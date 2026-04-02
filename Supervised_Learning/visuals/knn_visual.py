"""
Self-contained HTML visual for K-Nearest Neighbours.
5 interactive tabs: Intro & How KNN Works, k Value & Decision Boundary,
Distance Metrics, Curse of Dimensionality, Weighted KNN & Normalisation.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(KNN_VISUAL_HTML, height=KNN_VISUAL_HEIGHT, scrolling=True)
"""

KNN_VISUAL_HTML = r"""<!DOCTYPE html>
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
  return 'rgba('+r+','+g+','+b+','+a+')';
}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function card(inner,extra){
  return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
}
function sectionTitle(t,s){
  return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
}
function insight(icon,title,body){
  return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx
    +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
}
function sliderRow(action,val,min,max,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
  var xm=xmax||10, ym=ymax||10;
  var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
  var o='';
  xts.forEach(function(v){
    o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  yts.forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  xts.forEach(function(v){
    o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  yts.forEach(function(v){
    o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}

/* ─── STATE ─── */
var S={
  tab:0,
  introK:3,       /* intro: k for worked example */
  introQuery:0,   /* intro: which query point (0..2) */
  boundaryK:3,    /* boundary tab: k slider 1..15 */
  distMetric:0,   /* distance tab: 0=euclidean,1=manhattan,2=chebyshev */
  dims:2,         /* curse of dim: 1..10 */
  wgtK:5,         /* weighted tab: k slider */
  wgtScale:0      /* weighted tab: 0=unscaled,1=scaled */
};

/* ══════════════════════════════════════════════════════════
   SHARED DATASET — 24 points, 3 classes
══════════════════════════════════════════════════════════ */
var PTS=[
  /* class 0 (blue) — top-left cluster */
  {x:1.5,y:8.2,c:0},{x:2.4,y:7.5,c:0},{x:1.8,y:9.0,c:0},{x:3.1,y:8.6,c:0},
  {x:2.0,y:6.8,c:0},{x:3.5,y:7.2,c:0},{x:1.2,y:7.8,c:0},{x:4.0,y:8.9,c:0},
  /* class 1 (orange) — bottom-right cluster */
  {x:7.0,y:1.8,c:1},{x:8.2,y:2.5,c:1},{x:6.5,y:3.2,c:1},{x:9.0,y:1.5,c:1},
  {x:7.8,y:3.8,c:1},{x:8.5,y:2.0,c:1},{x:6.2,y:2.2,c:1},{x:9.2,y:3.1,c:1},
  /* class 2 (purple) — middle cluster */
  {x:4.5,y:5.0,c:2},{x:5.5,y:5.8,c:2},{x:4.8,y:4.2,c:2},{x:6.1,y:5.2,c:2},
  {x:5.2,y:6.5,c:2},{x:3.8,y:4.8,c:2},{x:6.4,y:4.5,c:2},{x:5.0,y:3.5,c:2}
];
var PT_COLS=[C.blue,C.orange,C.purple];
var PT_NAMES=['Class A','Class B','Class C'];

/* 3 query points for Tab 0 */
var QUERY_PTS=[
  {x:3.2,y:6.0,label:'Q1'},
  {x:5.8,y:4.0,label:'Q2'},
  {x:7.5,y:5.5,label:'Q3'}
];

function euclidean(a,b){
  return Math.sqrt(Math.pow(a.x-b.x,2)+Math.pow(a.y-b.y,2));
}

function knnClassify(qx,qy,k,metric){
  var q={x:qx,y:qy};
  var dists=PTS.map(function(p,i){
    var d;
    if(metric===1){d=Math.abs(p.x-q.x)+Math.abs(p.y-q.y);}
    else if(metric===2){d=Math.max(Math.abs(p.x-q.x),Math.abs(p.y-q.y));}
    else{d=euclidean(p,q);}
    return {d:d,c:p.c,i:i};
  });
  dists.sort(function(a,b){return a.d-b.d;});
  var neighbors=dists.slice(0,k);
  var votes=[0,0,0];
  neighbors.forEach(function(n){votes[n.c]++;});
  var maxV=Math.max.apply(null,votes);
  return {cls:votes.indexOf(maxV),votes:votes,neighbors:neighbors};
}

/* ══════════════════════════════════════════════════════════
   TAB 0 — INTRO & HOW KNN WORKS
══════════════════════════════════════════════════════════ */
function renderIntro(){
  var k=S.introK;
  var qi=S.introQuery;
  var qp=QUERY_PTS[qi];
  var result=knnClassify(qp.x,qp.y,k,0);
  var neighbors=result.neighbors;

  /* ─── scatter plot ─── */
  var sv=plotAxes('Feature 1','Feature 2');

  /* decision region shading (coarse grid) */
  var step=0.5;
  for(var gx=0;gx<10;gx+=step){
    for(var gy=0;gy<10;gy+=step){
      var cx=gx+step/2, cy=gy+step/2;
      var pred=knnClassify(cx,cy,k,0).cls;
      sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
        +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
        +' fill="'+hex(PT_COLS[pred],0.07)+'"/>';
    }
  }

  /* training points */
  PTS.forEach(function(p,i){
    var isNeighbor=neighbors.some(function(n){return n.i===i;});
    var nIdx=neighbors.findIndex(function(n){return n.i===i;});
    var col=PT_COLS[p.c];
    if(isNeighbor){
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="9"'
        +' fill="none" stroke="'+col+'" stroke-width="2" stroke-dasharray="4,2" opacity="0.7"/>';
    }
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5"'
      +' fill="'+col+'" stroke="'+(isNeighbor?'#e4e4e7':'#0a0a0f')+'" stroke-width="'+(isNeighbor?2:1.5)+'"/>';
    if(isNeighbor){
      sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-11).toFixed(1)+'" text-anchor="middle"'
        +' fill="'+C.yellow+'" font-size="8" font-weight="700">'+(nIdx+1)+'</text>';
    }
  });

  /* lines to k nearest neighbors */
  neighbors.forEach(function(n){
    var p=PTS[n.i];
    sv+='<line x1="'+sx(qp.x).toFixed(1)+'" y1="'+sy(qp.y).toFixed(1)+'"'
      +' x2="'+sx(p.x).toFixed(1)+'" y2="'+sy(p.y).toFixed(1)+'"'
      +' stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>';
  });

  /* radius circle = dist to k-th neighbor */
  var kthDist=neighbors[k-1].d;
  var kthPx=kthDist*(PW/10);
  sv+='<circle cx="'+sx(qp.x).toFixed(1)+'" cy="'+sy(qp.y).toFixed(1)+'" r="'+kthPx.toFixed(1)+'"'
    +' fill="none" stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="5,4" opacity="0.4"/>';

  /* query point */
  var predCol=PT_COLS[result.cls];
  sv+='<circle cx="'+sx(qp.x).toFixed(1)+'" cy="'+sy(qp.y).toFixed(1)+'" r="7"'
    +' fill="'+predCol+'" stroke="'+C.yellow+'" stroke-width="2.5"/>';
  sv+='<text x="'+sx(qp.x).toFixed(1)+'" y="'+(sy(qp.y)-10).toFixed(1)+'" text-anchor="middle"'
    +' fill="'+C.yellow+'" font-size="10" font-weight="700">'+qp.label+'</text>';

  var out=sectionTitle('How KNN Works','Find the k nearest training examples by distance — predict the majority class');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  QUERY_PTS.forEach(function(qpt,i){
    out+=btnSel(i,qi,C.yellow,qpt.label+' ('+qpt.x+','+qpt.y+')','introQuery');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','k='+k+' Nearest Neighbours of '+qp.label)
    +svgBox(sv)
    +sliderRow('introK',k,1,15,1,'k',0)
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">'
    +[{c:C.blue,l:'Class A'},{c:C.orange,l:'Class B'},{c:C.purple,l:'Class C'},
      {c:C.yellow,l:'Query / neighbors'},{c:C.yellow,l:'Radius to k-th'}].map(function(it,idx){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
        +(idx<3?'<div style="width:10px;height:10px;border-radius:50%;background:'+it.c+'"></div>'
          :'<div style="width:10px;height:2px;background:'+it.c+';border-top:1px dashed '+it.c+'"></div>')
        +it.l+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','PREDICTION FOR '+qp.label);
  out+=statRow('k',k,C.accent);
  out+=statRow('Predicted class',PT_NAMES[result.cls],PT_COLS[result.cls]);
  result.votes.forEach(function(v,ci){
    out+=statRow(PT_NAMES[ci]+' votes',v,PT_COLS[ci]);
  });
  out+=statRow('k-th dist.',neighbors[k-1].d.toFixed(3),C.yellow);
  out+=statRow('Nearest dist.',neighbors[0].d.toFixed(3),C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE ALGORITHM (INFERENCE)');
  [
    {n:'1',lbl:'Compute distances',   desc:'d(x\u1D50, x\u1D35) for every training point',c:C.blue},
    {n:'2',lbl:'Sort & take top k',   desc:'Select k smallest distances',               c:C.orange},
    {n:'3',lbl:'Vote (or average)',    desc:'Classification: majority vote',              c:C.purple},
    {n:'4',lbl:'Return prediction',   desc:'Class with most votes wins',                 c:C.green},
  ].forEach(function(s){
    out+='<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="width:18px;height:18px;border-radius:50%;background:'+hex(s.c,0.2)+';border:1px solid '+s.c
      +';display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:9px;font-weight:800;color:'+s.c+';">'+s.n+'</div>'
      +'<div><div style="font-size:9.5px;font-weight:700;color:'+s.c+';">'+s.lbl+'</div>'
      +'<div style="font-size:8.5px;color:'+C.muted+';margin-top:1px;">'+s.desc+'</div></div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY PROPERTIES');
  [
    {icon:'\u2714',lbl:'Non-parametric (no training)',c:C.green},
    {icon:'\u2714',lbl:'Works for any class count',  c:C.green},
    {icon:'\u2714',lbl:'Naturally multi-class',      c:C.green},
    {icon:'\u2718',lbl:'Slow at inference: O(nd)',   c:C.red},
    {icon:'\u2718',lbl:'All training data in memory',c:C.red},
    {icon:'\u26a0',lbl:'Sensitive to scale & k',     c:C.yellow},
  ].forEach(function(r){
    out+='<div style="display:flex;gap:6px;padding:3px 0;font-size:9px;">'
      +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
      +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128269;','Lazy Learning: No Training Phase',
    'KNN is a <span style="color:'+C.yellow+';font-weight:700;">lazy learner</span> — it stores all training data and defers all computation to inference time. '
    +'Training cost: <span style="color:'+C.green+';font-family:monospace;">O(1)</span>. '
    +'Inference cost: <span style="color:'+C.red+';font-family:monospace;">O(nd)</span> per query (n=training size, d=dimensions). '
    +'Production systems use <span style="color:'+C.accent+';font-weight:700;">KD-trees or Ball-trees</span> to reduce this to '
    +'<span style="color:'+C.green+';font-family:monospace;">O(d\u00b7log n)</span> in low dimensions.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 1 — k VALUE & DECISION BOUNDARY
══════════════════════════════════════════════════════════ */
function knnBoundaryAcc(k){
  /* leave-one-out accuracy estimate */
  var correct=0;
  PTS.forEach(function(p,i){
    var dists=PTS.map(function(q,j){
      if(j===i) return {d:999,c:q.c};
      return {d:euclidean(p,q),c:q.c};
    });
    dists.sort(function(a,b){return a.d-b.d;});
    var votes=[0,0,0];
    dists.slice(0,k).forEach(function(n){votes[n.c]++;});
    var maxV=Math.max.apply(null,votes);
    if(votes.indexOf(maxV)===p.c) correct++;
  });
  return correct/PTS.length;
}

function renderBoundary(){
  var k=S.boundaryK;
  var acc=knnBoundaryAcc(k);

  /* ─── decision boundary plot ─── */
  var sv=plotAxes('Feature 1','Feature 2');
  var step=0.4;
  for(var gx=0;gx<10;gx+=step){
    for(var gy=0;gy<10;gy+=step){
      var cx=gx+step/2, cy=gy+step/2;
      var pred=knnClassify(cx,cy,k,0).cls;
      sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
        +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
        +' fill="'+hex(PT_COLS[pred],0.09)+'"/>';
    }
  }

  /* all training points */
  PTS.forEach(function(p){
    /* check if correctly classified (leave-one-out) */
    var dists=PTS.map(function(q,j){
      return {d:euclidean(p,q),c:q.c,j:j};
    }).filter(function(d){return d.d>0;});
    dists.sort(function(a,b){return a.d-b.d;});
    var votes=[0,0,0];
    dists.slice(0,k).forEach(function(n){votes[n.c]++;});
    var pred=votes.indexOf(Math.max.apply(null,votes));
    var correct=(pred===p.c);
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5"'
      +' fill="'+PT_COLS[p.c]+'" stroke="'+(correct?'#0a0a0f':C.red)+'" stroke-width="'+(correct?1.5:2.5)+'"/>';
    if(!correct){
      sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="9">\u2717</text>';
    }
  });

  /* accuracy badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="138" height="20" rx="4" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+15)+'" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">LOO acc: '+(acc*100).toFixed(0)+'%</text>';

  /* ─── k vs accuracy curve ─── */
  var AW=440, AH=110;
  var aPL=40, aPR=12, aPT=12, aPB=28;
  var aPW=AW-aPL-aPR, aPH=AH-aPT-aPB;
  var av='';
  /* grid */
  [0.6,0.7,0.8,0.9,1.0].forEach(function(v){
    var py=aPT+aPH-(v-0.5)/(0.5)*aPH;
    av+='<line x1="'+aPL+'" y1="'+py.toFixed(1)+'" x2="'+(aPL+aPW)+'" y2="'+py.toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    av+='<text x="'+(aPL-4)+'" y="'+(py+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+(v*100).toFixed(0)+'</text>';
  });
  [1,3,5,7,9,11,13,15].forEach(function(kv){
    var px=aPL+(kv-1)/14*aPW;
    av+='<text x="'+px.toFixed(1)+'" y="'+(aPT+aPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8">'+kv+'</text>';
  });
  av+='<line x1="'+aPL+'" y1="'+aPT+'" x2="'+aPL+'" y2="'+(aPT+aPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  av+='<line x1="'+aPL+'" y1="'+(aPT+aPH)+'" x2="'+(aPL+aPW)+'" y2="'+(aPT+aPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  av+='<text x="'+(aPL+aPW/2)+'" y="'+(AH-2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">k</text>';
  av+='<text x="8" y="'+(aPT+aPH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" transform="rotate(-90,8,'+(aPT+aPH/2)+')">Acc %</text>';

  /* acc curve */
  var accPath='';
  for(var kv=1;kv<=15;kv++){
    var accV=knnBoundaryAcc(kv);
    var px2=aPL+(kv-1)/14*aPW;
    var py2=aPT+aPH-(accV-0.5)/0.5*aPH;
    accPath+=(kv===1?'M':'L')+px2.toFixed(1)+','+py2.toFixed(1)+' ';
  }
  av+='<path d="'+accPath+'" fill="none" stroke="'+C.accent+'" stroke-width="2"/>';

  /* highlight sweet spot k=3..7 */
  var swL=aPL+2/14*aPW, swR=aPL+6/14*aPW;
  av+='<rect x="'+swL.toFixed(1)+'" y="'+aPT+'" width="'+(swR-swL).toFixed(1)+'" height="'+aPH+'" fill="'+hex(C.green,0.06)+'"/>';
  av+='<text x="'+(swL+(swR-swL)/2).toFixed(1)+'" y="'+(aPT+9)+'" text-anchor="middle" fill="'+C.green+'" font-size="7.5">sweet spot</text>';

  /* current k marker */
  var ckx=aPL+(k-1)/14*aPW;
  var ckAcc=knnBoundaryAcc(k);
  var cky=aPT+aPH-(ckAcc-0.5)/0.5*aPH;
  av+='<line x1="'+ckx.toFixed(1)+'" y1="'+aPT+'" x2="'+ckx.toFixed(1)+'" y2="'+(aPT+aPH)+'" stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="3,2"/>';
  av+='<circle cx="'+ckx.toFixed(1)+'" cy="'+cky.toFixed(1)+'" r="4" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  var complexity=k<=2?'Overfit (complex boundary)':k<=7?'Good balance':k<=11?'Smoothing':'Underfit (too smooth)';
  var complexCol=k<=2?C.red:k<=7?C.green:k<=11?C.yellow:C.orange;

  var out=sectionTitle('k Value & Decision Boundary','Small k = complex boundary (overfit); large k = smooth boundary (underfit)');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Decision Boundary (k='+k+')')
    +svgBox(sv)
    +'<div style="margin-top:8px;">'+svgBox(av,AW,AH)+'</div>'
    +sliderRow('boundaryK',k,1,15,1,'k',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.red+';">\u2190 k=1 (overfit)</span>'
    +'<span style="color:'+C.orange+';">k=15 (underfit) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','k='+k+' STATS');
  out+=statRow('LOO accuracy',(acc*100).toFixed(0)+'%',acc>0.85?C.green:acc>0.7?C.yellow:C.red);
  out+=statRow('Complexity',complexity,complexCol);
  var misclass=PTS.filter(function(p){
    var dists2=PTS.map(function(q,j){return {d:euclidean(p,q),c:q.c,j:j};}).filter(function(d){return d.d>0;});
    dists2.sort(function(a,b){return a.d-b.d;});
    var v2=[0,0,0]; dists2.slice(0,k).forEach(function(n){v2[n.c]++;});
    return v2.indexOf(Math.max.apply(null,v2))!==p.c;
  }).length;
  out+=statRow('Misclassified',misclass+' / '+PTS.length,misclass>5?C.red:C.yellow);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','CHOOSING k');
  [
    {k:'k=1',   desc:'Exact fit, noisy boundary, overfit on small data',     c:C.red},
    {k:'k=3..7',desc:'Sweet spot for most datasets',                          c:C.green},
    {k:'k=sqrt(n)',desc:'Rule of thumb: k\u2248\u221an training samples',    c:C.accent},
    {k:'k=n',   desc:'Always predicts majority class (underfit)',              c:C.orange},
  ].forEach(function(r){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9.5px;font-weight:700;font-family:monospace;color:'+r.c+';">'+r.k+'</div>'
      +'<div style="font-size:8.5px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','CHOOSE k BY CROSS-VALIDATION');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    '1. Try odd k values (avoids ties in binary problems)<br>'
    +'2. Use 5- or 10-fold CV on each k<br>'
    +'3. Pick k with best val accuracy<br>'
    +'4. Large datasets: restrict to <span style="color:'+C.accent+';">k &lt; 50</span><br>'
    +'5. sklearn: <span style="color:'+C.accent+';font-family:monospace;">GridSearchCV</span>'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9935;','k=1 is Bayes-Optimal Only with Infinite Data',
    'With n\u2192\u221e, the k=1 nearest neighbour is always the <em>identical</em> point, achieving Bayes error. '
    +'In practice k=1 memorises noise. '
    +'The <span style="color:'+C.yellow+';font-weight:700;">bias-variance tradeoff</span> is directly controlled by k: '
    +'small k = low bias / high variance; large k = high bias / low variance. '
    +'Use cross-validation to find the k that minimises generalisation error.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 2 — DISTANCE METRICS
══════════════════════════════════════════════════════════ */
var METRIC_NAMES=['Euclidean (L\u2082)','Manhattan (L\u2081)','Chebyshev (L\u221e)'];
var METRIC_COLS=[C.blue,C.orange,C.purple];

function renderDistance(){
  var mi=S.distMetric;
  /* query point */
  var qx=5.0, qy=5.0;

  /* ─── iso-distance contour plot ─── */
  /* show rings of equal distance=2 from query for each metric */
  var sv=plotAxes('Feature 1','Feature 2');
  /* draw all three contours faintly, highlight current */
  var contours=[
    /* Euclidean: circle */
    function(d){
      var pts=[];
      for(var angle=0;angle<360;angle+=5){
        var rad=angle*Math.PI/180;
        pts.push({x:qx+d*Math.cos(rad),y:qy+d*Math.sin(rad)});
      }
      return pts;
    },
    /* Manhattan: rotated square (diamond) */
    function(d){
      return [{x:qx,y:qy+d},{x:qx+d,y:qy},{x:qx,y:qy-d},{x:qx-d,y:qy},{x:qx,y:qy+d}];
    },
    /* Chebyshev: axis-aligned square */
    function(d){
      return [{x:qx-d,y:qy-d},{x:qx+d,y:qy-d},{x:qx+d,y:qy+d},{x:qx-d,y:qy+d},{x:qx-d,y:qy-d}];
    }
  ];
  [2.0,2.5].forEach(function(dist){
    [0,1,2].forEach(function(ci){
      var pts=contours[ci](dist);
      var path=pts.map(function(p,pi){
        return (pi===0?'M':'L')+sx(p.x).toFixed(1)+','+sy(p.y).toFixed(1);
      }).join(' ')+' Z';
      sv+='<path d="'+path+'" fill="none"'
        +' stroke="'+METRIC_COLS[ci]+'" stroke-width="'+(ci===mi?2.5:1)+'"'
        +' stroke-dasharray="'+(ci===mi?'':'6,4')+'" opacity="'+(ci===mi?0.9:0.3)+'"/>';
    });
  });

  /* dist=2.0 label */
  sv+='<text x="'+sx(qx+2.1).toFixed(1)+'" y="'+sy(qy+0.15).toFixed(1)+'" fill="'+METRIC_COLS[mi]+'" font-size="8.5" font-family="monospace">d=2</text>';
  sv+='<text x="'+sx(qx+2.6).toFixed(1)+'" y="'+sy(qy+0.15).toFixed(1)+'" fill="'+METRIC_COLS[mi]+'" font-size="8.5" font-family="monospace">d=2.5</text>';

  /* training points — coloured by class */
  PTS.forEach(function(p){
    var d;
    if(mi===1) d=Math.abs(p.x-qx)+Math.abs(p.y-qy);
    else if(mi===2) d=Math.max(Math.abs(p.x-qx),Math.abs(p.y-qy));
    else d=euclidean(p,{x:qx,y:qy});
    var close=d<=2.5;
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5"'
      +' fill="'+PT_COLS[p.c]+'" stroke="'+(close?'#e4e4e7':'#0a0a0f')+'" stroke-width="'+(close?2:1.5)+'"/>';
  });

  /* query point */
  sv+='<circle cx="'+sx(qx).toFixed(1)+'" cy="'+sy(qy).toFixed(1)+'" r="7"'
    +' fill="'+C.yellow+'" stroke="'+C.text+'" stroke-width="2"/>';
  sv+='<text x="'+sx(qx).toFixed(1)+'" y="'+(sy(qy)-11).toFixed(1)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="9" font-weight="700">Query</text>';

  /* ─── formula + per-point distance table ─── */
  var sample=[PTS[0],PTS[8],PTS[16]]; /* one from each class */
  var formulaStr=[
    'd(a,b) = \u221a\u03a3(a\u1d35\u2212b\u1d35)\u00b2',
    'd(a,b) = \u03a3|a\u1d35\u2212b\u1d35|',
    'd(a,b) = max|a\u1d35\u2212b\u1d35|'
  ];

  var out=sectionTitle('Distance Metrics','The shape of the neighbourhood changes which points are "nearest"');
  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  METRIC_NAMES.forEach(function(nm,i){
    out+=btnSel(i,mi,METRIC_COLS[i],nm,'distMetric');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Equal-distance Contours from Query (5,5)')
    +svgBox(sv)
    +'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    +METRIC_NAMES.map(function(nm,i){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+(i===mi?C.text:C.muted)+';">'
        +'<div style="width:14px;height:2px;background:'+METRIC_COLS[i]+'"></div>'+nm+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','FORMULA: '+METRIC_NAMES[mi]);
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';'
    +'font-size:10px;color:'+METRIC_COLS[mi]+';font-family:monospace;',formulaStr[mi]);
  out+='<div style="margin-top:10px;">';
  out+=div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','Sample distances from ('+qx+','+qy+'):');
  sample.forEach(function(p){
    var de=euclidean(p,{x:qx,y:qy});
    var dm=Math.abs(p.x-qx)+Math.abs(p.y-qy);
    var dc=Math.max(Math.abs(p.x-qx),Math.abs(p.y-qy));
    var vals=[de,dm,dc];
    out+='<div style="display:flex;justify-content:space-between;font-size:9px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+PT_COLS[p.c]+';">'+PT_NAMES[p.c]+' ('+p.x+','+p.y+')</span>'
      +'<span style="color:'+METRIC_COLS[mi]+';font-weight:700;font-family:monospace;">'+vals[mi].toFixed(2)+'</span></div>';
  });
  out+='</div></div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','METRIC GUIDE');
  [
    {m:'Euclidean',  pro:'Isotropic, matches intuition',    con:'Sensitive to scale',          c:C.blue},
    {m:'Manhattan',  pro:'Robust to outliers, sparse data', con:'Diagonal distance penalised',  c:C.orange},
    {m:'Chebyshev',  pro:'Warehouse routing, games',         con:'Unusual in ML',                c:C.purple},
    {m:'Cosine',     pro:'NLP, high-dim document vectors',   con:'Ignores magnitude',            c:C.green},
    {m:'Hamming',    pro:'Categorical / binary features',    con:'Numeric features only Eucl.',  c:C.yellow},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:2px 0;border-radius:4px;border-left:2px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.m+'</div>'
      +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
      +'<div style="font-size:8px;color:'+C.red+';">\u2718 '+r.con+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128207;','Metric Choice Shapes Your Neighbourhood',
    'The metric determines <em>which points count as neighbours</em>. '
    +'Euclidean is the default but only makes sense if all features are on the <span style="color:'+C.yellow+';font-weight:700;">same scale</span>. '
    +'Manhattan is preferred when features are <span style="color:'+C.accent+';font-weight:700;">independent and differently-scaled</span>. '
    +'Minkowski p is a generalisation: p=2\u2192Euclidean, p=1\u2192Manhattan, p\u2192\u221e\u2192Chebyshev. '
    +'Always <span style="color:'+C.green+';font-weight:700;">normalise features</span> before computing distances.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 3 — CURSE OF DIMENSIONALITY
══════════════════════════════════════════════════════════ */
function renderCurse(){
  var d=S.dims;

  /* fraction of unit hypercube volume in a ball of radius 0.5 */
  /* V_ball(r,d) = pi^(d/2) * r^d / Gamma(d/2+1) */
  /* we just use the ratio conceptually: fraction of data within r=0.5 */
  /* avg nearest-neighbour distance in d dims (n=500 uniform): ~ d^0.5 * Gamma(1+1/d) / (2*n^(1/d)) */
  function avgNNDist(dims){
    return Math.pow(1/500, 1/dims) * Math.pow(dims, 0.5) * 0.6;
  }
  function fractionWithinR(dims, r){
    /* fraction of points in sphere of radius r in d-dim unit hypercube */
    /* approximate: V_sphere / V_cube, capped at 1 */
    var pi=Math.PI;
    /* log gamma via Stirling */
    function lgamma(z){
      if(z<0.5) return Math.log(pi/Math.sin(pi*z))-lgamma(1-z);
      z-=1;
      var x=0.99999999999980993;
      var g=7;
      var c=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
      for(var i=0;i<g;i++) x+=c[i]/(z+i+1);
      var t=z+g+0.5;
      return 0.5*Math.log(2*pi)+(z+0.5)*Math.log(t)-t+Math.log(x);
    }
    var logVball=dims/2*Math.log(pi)+dims*Math.log(r)-lgamma(dims/2+1);
    var frac=Math.exp(logVball);
    return Math.min(frac,1);
  }

  var nnDist=avgNNDist(d);
  var frac=fractionWithinR(d, 0.5);
  var expectedNeighborsInR=Math.round(frac*500);

  /* ─── sparsity visualisation ─── */
  /* Show: in d dimensions, the fraction of volume in outer shell (radius 0.9 to 1.0) */
  /* fraction in outer 10% shell = 1 - 0.9^d */
  function shellFrac(dims){ return 1 - Math.pow(0.9, dims); }

  /* ─── bar chart: volume fraction in outer shell by dimension ─── */
  var BW=440, BH=160;
  var bPL=30, bPR=12, bPT=12, bPB=28;
  var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
  var dims=[1,2,3,5,8,10,20,50];
  var sv2='';
  /* axes */
  sv2+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv2+='<line x1="'+bPL+'" y1="'+(bPT+bPH)+'" x2="'+(bPL+bPW)+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,0.25,0.5,0.75,1.0].forEach(function(v){
    var py=bPT+bPH-v*bPH;
    sv2+='<line x1="'+bPL+'" y1="'+py.toFixed(1)+'" x2="'+(bPL+bPW)+'" y2="'+py.toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv2+='<text x="'+(bPL-3)+'" y="'+(py+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8">'+(v*100).toFixed(0)+'%</text>';
  });
  sv2+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5">Dimensions</text>';
  sv2+='<text x="8" y="'+(bPT+bPH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" transform="rotate(-90,8,'+(bPT+bPH/2)+')">Outer shell %</text>';

  var barW2=(bPW/dims.length)-4;
  dims.forEach(function(dv,i){
    var sf=shellFrac(dv);
    var bh=sf*bPH;
    var bx=bPL+4+i*(barW2+4);
    var by=bPT+bPH-bh;
    var active=(dv===d||(d>20&&i===dims.length-1));
    var col=sf>0.9?C.red:sf>0.6?C.orange:sf>0.3?C.yellow:C.green;
    sv2+='<rect x="'+bx.toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+barW2+'" height="'+bh.toFixed(1)+'" rx="2"'
      +' fill="'+hex(col,0.75)+'" stroke="'+(active?C.text:col)+'" stroke-width="'+(active?1.5:0)+'"/>';
    sv2+='<text x="'+(bx+barW2/2).toFixed(1)+'" y="'+(bPT+bPH+12)+'" text-anchor="middle" fill="'+(active?C.text:C.muted)+'" font-size="8">'+dv+'</text>';
    if(sf>0.05){
      sv2+='<text x="'+(bx+barW2/2).toFixed(1)+'" y="'+(by+9).toFixed(1)+'" text-anchor="middle" fill="#0a0a0f" font-size="7.5" font-weight="700">'+(sf*100).toFixed(0)+'%</text>';
    }
  });

  var sf=shellFrac(d);
  var sfCol=sf>0.9?C.red:sf>0.6?C.orange:sf>0.3?C.yellow:C.green;

  var out=sectionTitle('Curse of Dimensionality','In high dimensions, all points are far away — nearest neighbours stop being meaningfully "near"');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Volume in Outer 10% Shell of Unit Hypersphere')
    +svgBox(sv2,BW,BH)
    +sliderRow('dims',d,1,50,1,'dimensions',0)
    +'<div style="margin-top:6px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
    +div('font-size:8.5px;color:'+C.muted+';','At d='+d+': <span style="color:'+sfCol+';font-weight:700;">'+(sf*100).toFixed(1)+'%</span>'
      +' of the volume lives in the outer shell \u2014 "near" becomes meaningless.')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','d='+d+' DIMENSIONS (n=500)');
  out+=statRow('Outer shell fraction',(sf*100).toFixed(1)+'%',sfCol);
  out+=statRow('Fraction near origin',(fractionWithinR(d,0.5)*100).toFixed(2)+'%',C.blue);
  out+=statRow('Expected neighbours in r=0.5',expectedNeighborsInR,expectedNeighborsInR>10?C.green:C.red);
  out+=statRow('Avg NN distance',nnDist.toFixed(3),nnDist>0.5?C.red:C.green);
  out+=statRow('Data needed to fill space','~10\u1d48',d>5?C.red:C.yellow);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY KNN DEGRADES');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'In d dimensions, the ratio of distances to the <span style="color:'+C.red+';font-weight:700;">furthest</span> vs '
    +'<span style="color:'+C.green+';font-weight:700;">nearest</span> neighbour \u2192 1 as d grows. '
    +'All points become equidistant \u2014 the k-nearest have no more predictive value '
    +'than any other point. '
    +'Beyond d\u224820\u201330, Euclidean KNN typically degrades severely.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','REMEDIES');
  [
    {lbl:'Feature selection',   desc:'Fewer, more informative features',    c:C.green},
    {lbl:'PCA / SVD',           desc:'Project to lower-dim subspace',       c:C.blue},
    {lbl:'Feature importance',  desc:'Weight dimensions by relevance',      c:C.purple},
    {lbl:'Cosine distance',     desc:'Better for sparse high-dim data',     c:C.accent},
    {lbl:'Use another model',   desc:'SVM, tree models scale better',       c:C.yellow},
  ].forEach(function(r){
    out+='<div style="display:flex;gap:6px;align-items:flex-start;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="width:6px;height:6px;border-radius:50%;background:'+r.c+';margin-top:3px;flex-shrink:0;"></div>'
      +'<div><div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';">'+r.desc+'</div></div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127760;','Exponential Data Hunger',
    'To maintain the <em>same density</em> of neighbours, data requirements grow <span style="color:'+C.red+';font-weight:700;">exponentially</span> with dimension. '
    +'Covering a 10-dimensional unit hypercube with the same density as 2D requires '
    +'<span style="color:'+C.red+';font-family:monospace;">10\u00b9\u2070 = 10 billion</span> points. '
    +'This is why KNN (and kernel methods) are largely replaced by tree-based models and neural networks for high-dimensional data. '
    +'<span style="color:'+C.accent+';font-weight:700;">Best practical limit: d &lt; 20</span>.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 4 — WEIGHTED KNN & NORMALISATION
══════════════════════════════════════════════════════════ */
/* Unscaled dataset: Feature 1 in [0,10], Feature 2 in [0,0.1] — scale dominates */
var PTS_RAW=[
  {x1:1.5,x2:0.075,c:0},{x1:2.0,x2:0.082,c:0},{x1:2.8,x2:0.068,c:0},{x1:3.2,x2:0.090,c:0},
  {x1:7.0,x2:0.018,c:1},{x1:7.8,x2:0.025,c:1},{x1:8.5,x2:0.030,c:1},{x1:6.5,x2:0.010,c:1},
  {x1:4.5,x2:0.050,c:2},{x1:5.2,x2:0.055,c:2},{x1:5.8,x2:0.042,c:2},{x1:4.8,x2:0.060,c:2}
];

function normalize(pts){
  var x1min=Math.min.apply(null,pts.map(function(p){return p.x1;}));
  var x1max=Math.max.apply(null,pts.map(function(p){return p.x1;}));
  var x2min=Math.min.apply(null,pts.map(function(p){return p.x2;}));
  var x2max=Math.max.apply(null,pts.map(function(p){return p.x2;}));
  return pts.map(function(p){
    return {
      x1:(p.x1-x1min)/(x1max-x1min)*10,
      x2:(p.x2-x2min)/(x2max-x2min)*10,
      c:p.c
    };
  });
}

function renderWeighted(){
  var k=S.wgtK;
  var scaled=S.wgtScale===1;
  var pts=scaled?normalize(PTS_RAW):PTS_RAW.map(function(p){return {x1:p.x1,x2:p.x2*100,c:p.c};});
  var qx=4.5, qy=scaled?5.0:0.045*100;

  /* distances in current scale */
  var dists=pts.map(function(p,i){
    var d=Math.sqrt(Math.pow(p.x1-qx,2)+Math.pow(p.x2-qy,2));
    return {d:d,c:p.c,i:i};
  });
  dists.sort(function(a,b){return a.d-b.d;});
  var neighbors=dists.slice(0,k);

  /* weighted vote: weight = 1/d^2 */
  var wvotes=[0,0,0];
  var uvotes=[0,0,0];
  neighbors.forEach(function(n){
    uvotes[n.c]++;
    wvotes[n.c]+=(n.d>0?1/(n.d*n.d):1e6);
  });
  var upred=uvotes.indexOf(Math.max.apply(null,uvotes));
  var wpred=wvotes.indexOf(Math.max.apply(null,wvotes));

  /* ─── scatter (normalised or raw x2 *100 for visibility) ─── */
  var xmax=scaled?11:11, ymax=scaled?11:5.5;
  var sv='';
  /* mini axes */
  sv+=plotAxes('F1',scaled?'F2 (scaled)':'F2 (raw\u00d7100)',xmax,ymax,
    scaled?[0,2,4,6,8,10]:[0,2,4,6,8,10],
    scaled?[0,2,4,6,8,10]:[0,1,2,3,4,5]);

  /* training points */
  pts.forEach(function(p,i){
    var isN=neighbors.some(function(n){return n.i===i;});
    var nd=neighbors.find(function(n){return n.i===i;});
    var w=nd?(1/(nd.d*nd.d)):0;
    var wNorm=Math.min(w/5,1);
    sv+='<circle cx="'+sx(p.x1,xmax).toFixed(1)+'" cy="'+sy(p.x2,ymax).toFixed(1)+'" r="'+(isN?6:4.5)+'"'
      +' fill="'+PT_COLS[p.c]+'" stroke="'+(isN?C.yellow:'#0a0a0f')+'" stroke-width="'+(isN?2:1.5)+'"/>';
    if(isN){
      sv+='<line x1="'+sx(qx,xmax).toFixed(1)+'" y1="'+sy(qy,ymax).toFixed(1)+'"'
        +' x2="'+sx(p.x1,xmax).toFixed(1)+'" y2="'+sy(p.x2,ymax).toFixed(1)+'"'
        +' stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>';
      sv+='<text x="'+sx(p.x1,xmax).toFixed(1)+'" y="'+(sy(p.x2,ymax)-9).toFixed(1)+'" text-anchor="middle"'
        +' fill="'+C.yellow+'" font-size="8" font-weight="700">w='+(nd.d>0?(1/(nd.d*nd.d)).toFixed(2):'inf')+'</text>';
    }
  });

  /* query */
  sv+='<circle cx="'+sx(qx,xmax).toFixed(1)+'" cy="'+sy(qy,ymax).toFixed(1)+'" r="7"'
    +' fill="'+(upred===wpred?PT_COLS[upred]:C.red)+'" stroke="'+C.text+'" stroke-width="2.5"/>';
  sv+='<text x="'+sx(qx,xmax).toFixed(1)+'" y="'+(sy(qy,ymax)-11).toFixed(1)+'" text-anchor="middle" fill="'+C.text+'" font-size="9" font-weight="700">Q</text>';

  var out=sectionTitle('Weighted KNN & Feature Normalisation','Closer neighbours vote more; unscaled features make distance meaningless');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  out+=btnSel(0,S.wgtScale,C.red,'Unscaled features','wgtScale');
  out+=btnSel(1,S.wgtScale,C.green,'Normalised (min-max)','wgtScale');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',
      (scaled?'Normalised':'Unscaled')+' Features \u2014 Weighted KNN (k='+k+')')
    +svgBox(sv)
    +sliderRow('wgtK',k,1,10,1,'k',0)
    +'<div style="margin-top:8px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
    +div('font-size:8.5px;color:'+C.muted+';',
      scaled?'Both features now [0,10] \u2014 equally weighted in distance.'
            :'F1 range [0,10], F2 range [0,0.1]\u00d7100=[0,5] \u2014 F1 dominates.')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','VOTE COMPARISON (k='+k+')');
  out+=div('font-size:10px;color:'+C.yellow+';margin-bottom:6px;','Uniform weights:');
  uvotes.forEach(function(v,ci){
    out+=statRow(PT_NAMES[ci],v+' votes',v===Math.max.apply(null,uvotes)?PT_COLS[ci]:C.muted);
  });
  out+=statRow('Prediction',PT_NAMES[upred],PT_COLS[upred]);
  out+=div('font-size:10px;color:'+C.accent+';margin-top:8px;margin-bottom:6px;','Distance-weighted (1/d\u00b2):');
  wvotes.forEach(function(v,ci){
    out+=statRow(PT_NAMES[ci],v.toFixed(2)+' wt',v===Math.max.apply(null,wvotes)?PT_COLS[ci]:C.muted);
  });
  out+=statRow('Prediction',PT_NAMES[wpred],PT_COLS[wpred]);
  out+=statRow('Agree?',upred===wpred?'Yes':'No (weighting matters!)',upred===wpred?C.green:C.red);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','NORMALISATION METHODS');
  [
    {m:'Min-Max scaling',   f:'(x\u2212min)/(max\u2212min)',   r:'[0,1]',        c:C.blue},
    {m:'Z-score (Std.)',    f:'(x\u2212\u03bc)/\u03c3',        r:'mean=0,std=1', c:C.orange},
    {m:'Max-Abs',           f:'x/|x|_max',                    r:'[\u22121,1]',  c:C.purple},
    {m:'RobustScaler',      f:'(x\u2212Q2)/(Q3\u2212Q1)',      r:'IQR-based',    c:C.green},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:2px 0;border-radius:4px;border-left:2px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.m+'</div>'
      +'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:1px;">'
      +'<span style="color:'+C.muted+';font-family:monospace;">'+r.f+'</span>'
      +'<span style="color:'+C.dim+';">'+r.r+'</span></div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','SKLEARN PIPELINE');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:8.5px;color:'+C.muted+';line-height:1.9;font-family:monospace;',
    '<span style="color:'+C.accent+';">from</span> sklearn.pipeline <span style="color:'+C.accent+';">import</span> Pipeline<br>'
    +'<span style="color:'+C.accent+';">from</span> sklearn.preprocessing <span style="color:'+C.accent+';">import</span> StandardScaler<br>'
    +'<span style="color:'+C.accent+';">from</span> sklearn.neighbors <span style="color:'+C.accent+';">import</span> KNeighborsClassifier<br><br>'
    +'pipe = Pipeline([<br>'
    +'&nbsp;&nbsp;(<span style="color:'+C.green+'">"scaler"</span>, StandardScaler()),<br>'
    +'&nbsp;&nbsp;(<span style="color:'+C.green+'">"knn"</span>, KNeighborsClassifier(<br>'
    +'&nbsp;&nbsp;&nbsp;&nbsp;n_neighbors='+k+', weights=<span style="color:'+C.green+'">"distance"</span>))<br>'
    +'])'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9878;&#65039;','Always Scale Before KNN',
    'Feature scaling is not optional for KNN \u2014 it is <span style="color:'+C.red+';font-weight:700;">essential</span>. '
    +'A feature with range [0, 10000] will dominate all distance calculations, '
    +'making other features irrelevant regardless of their predictive power. '
    +'<span style="color:'+C.accent+';font-weight:700;">StandardScaler</span> (z-score) is the safest default. '
    +'Use <span style="color:'+C.yellow+';font-weight:700;">distance-weighted KNN</span> '
    +'(<span style="color:'+C.accent+';font-family:monospace;">weights="distance"</span>) '
    +'to reduce the influence of the k-th neighbour when it is much further than the first.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════════ */
var TABS=[
  '&#128269; How KNN Works',
  '&#9935; k &amp; Boundary',
  '&#128207; Distance Metrics',
  '&#127760; Curse of Dim.',
  '&#9878;&#65039; Weighted &amp; Scale'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.blue+','+C.purple+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">K-Nearest Neighbours</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough \u2014 from lazy learning and k selection to distance metrics, dimensionality curse and feature scaling')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderBoundary();
  else if(S.tab===2) html+=renderDistance();
  else if(S.tab===3) html+=renderCurse();
  else if(S.tab===4) html+=renderWeighted();
  html+='</div></div>';
  return html;
}

function render(){
  document.getElementById('app').innerHTML=renderApp();
  bindEvents();
}

function bindEvents(){
  document.querySelectorAll('[data-action]').forEach(function(el){
    var action=el.getAttribute('data-action');
    var idx=parseInt(el.getAttribute('data-idx'));
    var tag=el.tagName.toLowerCase();
    if(tag==='button'){
      el.addEventListener('click',function(){
        if(action==='tab')            {S.tab=idx;            render();}
        else if(action==='introQuery'){S.introQuery=idx;     render();}
        else if(action==='distMetric'){S.distMetric=idx;     render();}
        else if(action==='wgtScale')  {S.wgtScale=idx;       render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='introK')         {S.introK=Math.round(val);     render();}
        else if(action==='boundaryK') {S.boundaryK=Math.round(val);  render();}
        else if(action==='dims')      {S.dims=Math.round(val);       render();}
        else if(action==='wgtK')      {S.wgtK=Math.round(val);       render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

KNN_VISUAL_HEIGHT = 1100