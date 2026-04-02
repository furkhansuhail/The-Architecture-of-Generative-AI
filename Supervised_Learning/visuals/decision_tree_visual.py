"""
Self-contained HTML visual for Decision Trees.
5 interactive tabs: Intro & Axis-Aligned Splits, Impurity Criteria (Gini vs Entropy),
CART Growing Process, Overfitting & Pruning, Feature Importance.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(DT_VISUAL_HTML, height=DT_VISUAL_HEIGHT, scrolling=True)
"""

DT_VISUAL_HTML = r"""<!DOCTYPE html>
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
  return '<svg width="100%" viewBox="0 0 '+(w||520)+' '+(h||340)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=520,VH=340,PL=52,PR=18,PT=18,PB=44;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
  var xm=xmax||10, ym=ymax||10;
  var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
  var o='';
  xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
  yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}

/* ─── STATE ─── */
var S={
  tab:0,
  dtDepth:1,      /* intro split depth 1..3 */
  pVal:0.5,       /* impurity p slider */
  cartStep:0,     /* 0..4 growing steps */
  treeDepth:1,    /* overfitting depth 1..8 */
  pruneAlpha:0,   /* ccp_alpha 0..0.5 */
  featDataset:0   /* 0=synthetic A  1=synthetic B */
};

/* ══════════════════════════════════════════════════════
   TAB 0 — INTRO & AXIS-ALIGNED SPLITS
══════════════════════════════════════════════════════ */
/* Fixed 2-class dataset with two clusters */
var INTRO_PTS=[
  /* class 0 (blue) — upper-left cluster */
  {x:1.5,y:7.5,c:0},{x:2.2,y:8.8,c:0},{x:3.0,y:7.0,c:0},{x:1.8,y:9.2,c:0},
  {x:2.8,y:6.2,c:0},{x:1.2,y:6.8,c:0},{x:3.5,y:8.5,c:0},{x:4.2,y:7.8,c:0},
  /* class 1 (orange) — lower-right cluster */
  {x:6.5,y:2.5,c:1},{x:7.8,y:3.8,c:1},{x:8.5,y:1.8,c:1},{x:7.2,y:1.2,c:1},
  {x:9.0,y:3.2,c:1},{x:6.0,y:4.5,c:1},{x:8.0,y:4.8,c:1},{x:9.5,y:2.0,c:1},
  /* overlap region */
  {x:4.8,y:5.5,c:0},{x:5.5,y:4.8,c:1},{x:5.2,y:6.2,c:0},{x:6.2,y:5.8,c:1}
];

/* Precomputed splits for depth 1,2,3 */
var SPLITS=[
  /* depth 1 */
  [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]}],
  /* depth 2 — add horizontal cut in right half */
  [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
   {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]}],
  /* depth 3 — refine left side too */
  [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
   {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]},
   {feat:'x\u2082',thresh:6.5,dir:'horiz',val:6.5, region:[0,5,0,10]}]
];

function classifyPoint(p,depth){
  /* route point through learned axis-aligned splits */
  if(depth===0) return 0;
  if(p.x<=5.0){
    if(depth<3) return 0;
    return p.y<=6.5?1:0;
  } else {
    if(depth<2) return 1;
    return p.y<=5.0?1:0;
  }
}

function regionColor(x,y,depth){
  if(depth===0) return hex(C.blue,0.07);
  if(x<=5.0){
    if(depth<3) return hex(C.blue,0.07);
    return y<=6.5?hex(C.orange,0.07):hex(C.blue,0.07);
  } else {
    if(depth<2) return hex(C.orange,0.07);
    return y<=5.0?hex(C.orange,0.07):hex(C.blue,0.07);
  }
}

function renderIntro(){
  var depth=S.dtDepth;
  var splits=SPLITS[depth-1];

  /* accuracy */
  var correct=INTRO_PTS.filter(function(p){return classifyPoint(p,depth)===p.c;}).length;
  var acc=(correct/INTRO_PTS.length*100).toFixed(0);
  var accCol=acc>90?C.green:acc>75?C.yellow:C.red;

  /* build SVG */
  var sv=plotAxes('Feature x\u2081','Feature x\u2082');

  /* region shading — sample grid */
  var step=1;
  for(var gx=0;gx<10;gx+=step){
    for(var gy=0;gy<10;gy+=step){
      var cx2=gx+step/2, cy2=gy+step/2;
      sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
         +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
         +' fill="'+regionColor(cx2,cy2,depth)+'"/>';
    }
  }

  /* split lines */
  var splitColors=[C.accent,C.yellow,C.purple];
  splits.forEach(function(sp,i){
    var col=splitColors[i];
    if(sp.dir==='vert'){
      var lx=sx(sp.val);
      sv+='<line x1="'+lx.toFixed(1)+'" y1="'+PT+'" x2="'+lx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
      sv+='<rect x="'+(lx-52)+'" y="'+(PT+3)+'" width="100" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
      sv+='<text x="'+lx.toFixed(1)+'" y="'+(PT+13)+'" text-anchor="middle" fill="'+col+'" font-size="11" font-family="monospace" font-weight="700">x\u2081 \u2264 '+sp.val+'?</text>';
    } else {
      var ly=sy(sp.val);
      var rx1=sx(sp.region[0]), rx2=sx(sp.region[1]);
      sv+='<line x1="'+rx1.toFixed(1)+'" y1="'+ly.toFixed(1)+'" x2="'+rx2.toFixed(1)+'" y2="'+ly.toFixed(1)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
      sv+='<rect x="'+(rx2-110)+'" y="'+(ly-16)+'" width="108" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
      sv+='<text x="'+(rx2-6)+'" y="'+(ly-5)+'" text-anchor="end" fill="'+col+'" font-size="11" font-family="monospace" font-weight="700">x\u2082 \u2264 '+sp.val+'?</text>';
    }
  });

  /* data points */
  INTRO_PTS.forEach(function(p){
    var pred=classifyPoint(p,depth);
    var ok=pred===p.c;
    var col=p.c===0?C.blue:C.orange;
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+col+'"'
       +' stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'"/>';
    if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="12">\u2717</text>';
  });

  /* accuracy badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-22)+'" width="120" height="18" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+PH-8)+'" fill="'+accCol+'" font-size="12" font-family="monospace" font-weight="700">accuracy: '+acc+'%</text>';

  var out=sectionTitle('What is a Decision Tree?','A hierarchy of axis-aligned yes/no questions that partitions feature space into regions');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left: interactive plot */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Axis-Aligned Splits')
    +svgBox(sv)
    +'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
    +[{c:C.blue,l:'Class 0'},{c:C.orange,l:'Class 1'},{c:C.red,l:'Misclassified'}].map(function(i){
      return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
        +'<div style="width:10px;height:10px;border-radius:50%;background:'+i.c+';"></div>'+i.l+'</div>';
    }).join('')
    +'</div>'
    +sliderRow('dtDepth',depth,1,3,1,'tree depth',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.yellow+';">\u2190 coarse (1 split)</span>'
    +'<span style="color:'+C.green+';">fine (3 splits) \u2192</span></div>'
  );
  out+='</div>';

  /* right: concepts */
  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT TREE');
  out+=statRow('Depth',depth,C.accent);
  out+=statRow('Splits applied',depth,C.yellow);
  out+=statRow('Leaf regions',depth+1,C.purple);
  out+=statRow('Accuracy',acc+'%',accCol);
  out+=statRow('Class 0 correct',INTRO_PTS.filter(function(p){return p.c===0&&classifyPoint(p,depth)===0;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===0;}).length,C.blue);
  out+=statRow('Class 1 correct',INTRO_PTS.filter(function(p){return p.c===1&&classifyPoint(p,depth)===1;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===1;}).length,C.orange);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY CONCEPTS');
  [
    {lbl:'No linearity assumption',c:C.green},
    {lbl:'Axis-aligned boundaries only',c:C.yellow},
    {lbl:'Greedy recursive splitting',c:C.blue},
    {lbl:'Each leaf = one prediction',c:C.purple},
    {lbl:'Depth controls complexity',c:C.orange},
  ].forEach(function(row){
    out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
      +'<div style="width:8px;height:8px;border-radius:2px;background:'+row.c+';flex-shrink:0;"></div>'
      +'<span style="color:'+C.muted+';">'+row.lbl+'</span></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VS LINEAR MODELS');
  [
    {lbl:'Linear / LR / SVM',eq:'f(x) = w\u00b7x + b',desc:'features contribute additively',c:C.blue},
    {lbl:'Decision Tree',eq:'f(x) = lookup leaf(x)',desc:'features contribute conditionally',c:C.accent},
  ].forEach(function(m){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:6px;border-left:3px solid '+m.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+m.c+';">'+m.lbl+'</div>'
      +'<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';margin:1px 0;">'+m.eq+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';">'+m.desc+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','Why Trees Can Model Anything',
    'Each additional depth level doubles the number of leaf regions. '
    +'A tree of depth d creates up to <span style="color:'+C.yellow+';font-family:monospace;">2\u1d48</span> rectangular regions — '
    +'together they can approximate <span style="color:'+C.accent+';font-weight:700;">any</span> decision boundary, '
    +'no matter how non-linear. The price: depth without constraint leads to <span style="color:'+C.red+';font-weight:700;">overfitting</span>.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 1 — IMPURITY CRITERIA (GINI VS ENTROPY)
══════════════════════════════════════════════════════ */
function gini(p){return 2*p*(1-p);}
function entropy(p){
  if(p<=0||p>=1) return 0;
  return -p*Math.log2(p)-(1-p)*Math.log2(1-p);
}

function renderImpurity(){
  var p=S.pVal;
  var gv=gini(p), ev=entropy(p);

  /* build impurity curve chart */
  var sv=plotAxes('p (fraction class 1)','Impurity',1,1,
    [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    [0,0.2,0.4,0.6,0.8,1.0]);

  /* shade under curves */
  var gPath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
  var ePath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
  var steps=100;
  for(var i=0;i<=steps;i++){
    var pp=i/steps;
    gPath+=' L'+sx(pp,1).toFixed(1)+','+sy(gini(pp),1).toFixed(1);
    ePath+=' L'+sx(pp,1).toFixed(1)+','+sy(entropy(pp),1).toFixed(1);
  }
  var baseY=sy(0,1);
  gPath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
  ePath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
  sv+='<path d="'+gPath+'" fill="'+hex(C.orange,0.12)+'"/>';
  sv+='<path d="'+ePath+'" fill="'+hex(C.blue,0.10)+'"/>';

  /* curve lines */
  var gLine='', eLine='';
  for(var j=0;j<=steps;j++){
    var pj=j/steps;
    var gpt=sx(pj,1).toFixed(1)+','+sy(gini(pj),1).toFixed(1);
    var ept=sx(pj,1).toFixed(1)+','+sy(entropy(pj),1).toFixed(1);
    gLine+=(j===0?'M':'L')+gpt+' ';
    eLine+=(j===0?'M':'L')+ept+' ';
  }
  sv+='<path d="'+gLine+'" fill="none" stroke="'+C.orange+'" stroke-width="2.2"/>';
  sv+='<path d="'+eLine+'" fill="none" stroke="'+C.blue+'" stroke-width="2.2"/>';

  /* vertical cursor */
  var cx3=sx(p,1);
  sv+='<line x1="'+cx3.toFixed(1)+'" y1="'+PT+'" x2="'+cx3.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  /* dots on curves */
  sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(gv,1).toFixed(1)+'" r="6.5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(ev,1).toFixed(1)+'" r="6.5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* value labels */
  var lx=cx3+8;
  sv+='<rect x="'+(lx-2)+'" y="'+(sy(gv,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
  sv+='<text x="'+lx+'" y="'+(sy(gv,1)+0)+'" fill="'+C.orange+'" font-size="11" font-family="monospace">Gini='+gv.toFixed(3)+'</text>';
  sv+='<rect x="'+(lx-2)+'" y="'+(sy(ev,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
  sv+='<text x="'+lx+'" y="'+(sy(ev,1)+0)+'" fill="'+C.blue+'" font-size="11" font-family="monospace">H='+ev.toFixed(3)+'</text>';

  /* p label */
  sv+='<text x="'+cx3.toFixed(1)+'" y="'+(PT+PH+26)+'" text-anchor="middle" fill="'+C.accent+'" font-size="11" font-weight="700">p='+p.toFixed(2)+'</text>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="138" height="36" rx="4" fill="#0a0a0f" opacity="0.88"/>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+13)+'" x2="'+(PL+30)+'" y2="'+(PT+13)+'" stroke="'+C.orange+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+16)+'" fill="'+C.orange+'" font-size="11" font-family="monospace">Gini = 2p(1\u2212p)</text>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+28)+'" x2="'+(PL+30)+'" y2="'+(PT+28)+'" stroke="'+C.blue+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+31)+'" fill="'+C.blue+'" font-size="11" font-family="monospace">H = \u2212p\u00b7log\u2082p\u2026</text>';

  /* ─── Gain demo: candidate split ─── */
  var nParent=10, nPos=Math.round(p*nParent), nNeg=nParent-nPos;
  var gParent=gini(p), hParent=entropy(p);
  /* split: left 5/10, right 5/10 */
  var nL=5, pL=Math.min(1,Math.max(0,Math.round(p*1.2*nL)/nL));
  var nR=5, pR=Math.min(1,Math.max(0,Math.round(p*0.8*nR)/nR));
  pL=Math.min(1,pL); pR=Math.min(1,pR);
  var wGini=(nL/nParent)*gini(pL)+(nR/nParent)*gini(pR);
  var wH   =(nL/nParent)*entropy(pL)+(nR/nParent)*entropy(pR);
  var gGain=Math.max(0,gParent-wGini);
  var igGain=Math.max(0,hParent-wH);
  var gainCol=gGain>0.1?C.green:gGain>0.02?C.yellow:C.red;

  var out=sectionTitle('Impurity Criteria','Gini and Entropy measure how mixed the class labels are at each node');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Impurity vs Class Proportion p')
    +svgBox(sv)
    +sliderRow('pVal',p,0.01,0.99,0.01,'p (class 1)',2)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.green+';">\u2190 pure class 0 (impurity = 0)</span>'
    +'<span style="color:'+C.green+';">pure class 1 \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  /* current impurity values */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT p = '+p.toFixed(2));
  out+=statRow('Gini impurity',''+gv.toFixed(4),C.orange);
  out+=statRow('Entropy H',''+ev.toFixed(4)+' bits',C.blue);
  out+=statRow('Is pure?',(p<=0.01||p>=0.99)?'Yes \u2713':'No',(p<=0.01||p>=0.99)?C.green:C.red);
  out+=statRow('Max impurity',(Math.abs(p-0.5)<0.05)?'Yes \u2248 0.5':'No',(Math.abs(p-0.5)<0.05)?C.yellow:C.muted);
  out+='</div>';

  /* formulas */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','FORMULAS');
  [
    {lbl:'Gini (binary)',    eq:'1 \u2212 p\u00b2 \u2212 (1\u2212p)\u00b2',  sub:'= 2p(1\u2212p)',  c:C.orange},
    {lbl:'Entropy (binary)', eq:'\u2212p\u00b7log\u2082p \u2212 (1\u2212p)\u00b7log\u2082(1\u2212p)', sub:'max at p=0.5: H=1 bit',c:C.blue},
    {lbl:'Gini general',     eq:'1 \u2212 \u2211\u2c7c p\u2c7c\u00b2',          sub:'k classes',         c:C.yellow},
  ].forEach(function(f){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+f.c+';">'
      +'<div style="font-size:8.5px;color:'+C.dim+';">'+f.lbl+'</div>'
      +'<div style="font-size:9.5px;font-family:monospace;color:'+f.c+';margin:2px 0;">'+f.eq+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';">'+f.sub+'</div></div>';
  });
  out+='</div>';

  /* information gain demo */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INFORMATION GAIN DEMO');
  out+='<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Parent: '+nPos+' pos / '+nNeg+' neg (p='+p.toFixed(2)+')</div>';
  [
    {lbl:'Gini(parent)',   val:gParent.toFixed(4), c:C.orange},
    {lbl:'Weighted Gini',  val:wGini.toFixed(4),   c:C.orange},
    {lbl:'Gini Gain',      val:gGain.toFixed(4),   c:gainCol},
    {lbl:'H(parent)',      val:hParent.toFixed(4)+' bits',c:C.blue},
    {lbl:'Info Gain (IG)', val:igGain.toFixed(4)+' bits', c:gainCol},
  ].forEach(function(r){out+=statRow(r.lbl,r.val,r.c);});
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128200;','Gini vs Entropy in Practice',
    'Gini is <span style="color:'+C.green+';font-weight:700;">slightly faster</span> (no log) and is the sklearn default. '
    +'Entropy is <span style="color:'+C.yellow+';font-weight:700;">more sensitive near pure nodes</span> and can prefer different splits on ambiguous data. '
    +'In practice, <span style="color:'+C.accent+';font-weight:700;">both produce nearly identical trees</span> — the choice rarely changes final accuracy by more than ~1%.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 2 — CART GROWING PROCESS
══════════════════════════════════════════════════════ */
var CART_STEPS=[
  {
    title:'Step 0 — All data at root',
    desc:'All 10 training examples land at the root. We compute the parent Gini impurity (6 Yes, 4 No \u2192 p=0.6).',
    detail:'Gini(parent) = 1 \u2212 0.6\u00b2 \u2212 0.4\u00b2 = 1 \u2212 0.36 \u2212 0.16 = 0.48',
    splitLabel:null,
    nodes:[
      {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true}
    ],
    edges:[]
  },
  {
    title:'Step 1 — Evaluate candidate splits',
    desc:'For each feature and threshold, compute weighted child Gini. Find the split with maximum Gain.',
    detail:'Age \u2264 42: Gain = 0.02 (small)\nIncome = High: Gain = 0.48 \u2714 (maximum!)',
    splitLabel:'Best split: Income = High',
    nodes:[
      {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true},
      {id:'cA',x:100,y:128,w:130,h:40,label:'Income = High',sub:'Gini = 0.48 (Age \u226442)',col:C.red,active:false,dim:true},
      {id:'cB',x:280,y:128,w:130,h:40,label:'Income = High',sub:'Gain = 0.48 \u2713 BEST',col:C.green,active:true},
    ],
    edges:[
      {from:'root',to:'cA',label:'Age\u226442',col:C.dim},
      {from:'root',to:'cB',label:'Income',col:C.green}
    ]
  },
  {
    title:'Step 2 — Apply best split',
    desc:'Split on Income = High. Left child (High): 6 Yes, 0 No — PURE! Right child (Low): 0 Yes, 4 No — PURE!',
    detail:'Weighted Gini = (6/10)\u00b70 + (4/10)\u00b70 = 0.00  \u2192  Gini Gain = 0.48',
    splitLabel:'Both children are pure leaves \u2713',
    nodes:[
      {id:'root',x:220,y:22,w:170,h:40,label:'Income = High?',sub:'Gain=0.48 | n=10',col:C.accent,active:false},
      {id:'left',x:100,y:120,w:140,h:46,label:'\u2714 High (Yes)',sub:'6 Yes, 0 No\nGini = 0.00 (PURE)',col:C.green,active:true},
      {id:'right',x:300,y:120,w:140,h:46,label:'\u2718 Low (No)',sub:'0 Yes, 4 No\nGini = 0.00 (PURE)',col:C.green,active:true},
    ],
    edges:[
      {from:'root',to:'left', label:'Yes (High)',col:C.green},
      {from:'root',to:'right',label:'No (Low)', col:C.orange}
    ]
  },
  {
    title:'Step 3 — Stopping conditions checked',
    desc:'Both children are pure (Gini = 0). Recursion stops. Leaves are assigned majority class predictions.',
    detail:'Stopping rule: node is pure \u2192 create leaf\nOther rules: max_depth reached, min_samples_split, zero gain',
    splitLabel:'Tree fully grown \u2014 depth = 1',
    nodes:[
      {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'root node | n=10',col:C.accent,active:false},
      {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES',sub:'Leaf | Gini=0 | n=6',col:C.green,active:true,leaf:true},
      {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | Gini=0 | n=4',col:C.blue,active:true,leaf:true},
    ],
    edges:[
      {from:'root',to:'left', label:'High',col:C.green},
      {from:'root',to:'right',label:'Low', col:C.orange}
    ]
  },
  {
    title:'Step 4 — Making a prediction',
    desc:'New example: Age=35, Income=High. Route from root \u2192 follow "High" branch \u2192 reach leaf \u2192 Predict: YES.',
    detail:'Runtime: O(depth) = O(1) for this tree.\nOnly the path root\u2192leaf is evaluated \u2014 all other nodes are ignored.',
    splitLabel:'Prediction in O(depth) time',
    nodes:[
      {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'New example: Income=High',col:C.yellow,active:true},
      {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES \u2714',sub:'New example lands here',col:C.green,active:true,leaf:true,highlight:true},
      {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | n=4',col:C.blue,active:false,leaf:true},
    ],
    edges:[
      {from:'root',to:'left', label:'\u2714 High \u2192 here',col:C.yellow},
      {from:'root',to:'right',label:'Low',col:C.dim}
    ]
  }
];

function treeNode(n){
  var fill=n.dim?hex(C.card,0.5):(n.highlight?hex(n.col,0.25):hex(n.col,0.12));
  var border=n.dim?C.dim:(n.active?n.col:C.border);
  var bw=n.active?2:1;
  var lines=n.sub.split('\n');
  return '<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'"'
    +' rx="7" fill="'+fill+'" stroke="'+border+'" stroke-width="'+bw+'"/>'
    +(n.leaf?'<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="4" rx="2" fill="'+border+'"/>':'')
    +'<text x="'+n.x+'" y="'+(n.y+14)+'" text-anchor="middle" fill="'+(n.dim?C.dim:n.col)+'" font-size="12" font-family="monospace" font-weight="700">'+n.label+'</text>'
    +lines.map(function(l,i){
      return '<text x="'+n.x+'" y="'+(n.y+27+i*11)+'" text-anchor="middle" fill="'+(n.dim?C.dim:C.muted)+'" font-size="12" font-family="monospace">'+l+'</text>';
    }).join('');
}

function treeEdge(e,nodes){
  var fromNode=nodes.find(function(n){return n.id===e.from;});
  var toNode=nodes.find(function(n){return n.id===e.to;});
  if(!fromNode||!toNode) return '';
  var x1=fromNode.x, y1=fromNode.y+fromNode.h;
  var x2=toNode.x, y2=toNode.y;
  var mx=(x1+x2)/2, my=(y1+y2)/2;
  return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+e.col+'" stroke-width="1.5"/>'
    +'<rect x="'+(mx-28)+'" y="'+(my-8)+'" width="56" height="12" rx="3" fill="#0a0a0f" opacity="0.85"/>'
    +'<text x="'+mx+'" y="'+(my+2)+'" text-anchor="middle" fill="'+e.col+'" font-size="11" font-family="monospace">'+e.label+'</text>';
}

function renderCART(){
  var step=S.cartStep;
  var cs=CART_STEPS[step];

  var svInner='';
  cs.edges.forEach(function(e){ svInner+=treeEdge(e,cs.nodes); });
  cs.nodes.forEach(function(n){ svInner+=treeNode(n); });

  var out=sectionTitle('CART Growing Process','The greedy recursive algorithm that builds decision trees one split at a time');

  /* step navigation */
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  CART_STEPS.forEach(function(st,i){
    var on=i===step;
    out+='<button data-action="cartStep" data-idx="'+i+'"'
      +' style="padding:7px 12px;border-radius:8px;font-size:9.5px;font-weight:700;font-family:inherit;'
      +'background:'+(on?hex(C.accent,.15):C.card)+';border:1.5px solid '+(on?C.accent:C.border)+';'
      +'color:'+(on?C.accent:C.muted)+';cursor:pointer;transition:all .2s;">'
      +'Step '+i+'</button>';
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* tree diagram */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',cs.title)
    +svgBox(svInner,520,240)
  );
  out+='</div>';

  /* step explanation */
  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:10px;','WHAT\'S HAPPENING');
  out+=div('font-size:9.5px;color:'+C.text+';line-height:1.8;margin-bottom:10px;',cs.desc);
  out+='<div style="background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';">'
    +'<pre style="font-size:8.5px;color:'+C.muted+';white-space:pre-wrap;line-height:1.7;font-family:monospace;">'+cs.detail+'</pre>'
    +'</div>';
  if(cs.splitLabel){
    out+='<div style="margin-top:10px;padding:8px 12px;background:'+hex(C.green,.08)+';border-radius:6px;border:1px solid '+hex(C.green,.3)+';font-size:9.5px;font-weight:700;color:'+C.green+';">'
      +'\u2714\ufe0f '+cs.splitLabel+'</div>';
  }
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','STOPPING CONDITIONS');
  [
    {c:C.green,  lbl:'Node is pure (Gini=0)', active:step>=2},
    {c:C.yellow, lbl:'max_depth reached',     active:false},
    {c:C.blue,   lbl:'min_samples_split',     active:false},
    {c:C.orange, lbl:'Zero gain split',       active:false},
  ].forEach(function(r){
    out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
      +'<div style="width:8px;height:8px;border-radius:50%;background:'+(r.active?r.c:C.dim)+';flex-shrink:0;"></div>'
      +'<span style="color:'+(r.active?r.c:C.dim)+';">'+r.lbl+(r.active?' \u2714':'')+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  /* navigation buttons */
  out+='<div style="display:flex;gap:10px;justify-content:center;margin-bottom:12px;">';
  out+='<button data-action="cartPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(step>0?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Prev</button>';
  out+='<button data-action="cartNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(step<4?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
    +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next \u2192</button>';
  out+='</div>';

  out+=insight('&#128296;','Why Greedy? Why Not Global?',
    'Finding the <span style="color:'+C.red+';font-weight:700;">globally optimal tree</span> is NP-hard — '
    +'the number of possible trees grows super-exponentially with depth and features. '
    +'CART\'s <span style="color:'+C.accent+';font-weight:700;">greedy approach</span> makes locally optimal decisions at each node in O(np log n) time. '
    +'The greedy solution is good enough in practice — '
    +'and ensembles like <span style="color:'+C.yellow+';font-weight:700;">Random Forests</span> compensate for its suboptimality.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 3 — OVERFITTING & PRUNING
══════════════════════════════════════════════════════ */
/* Simulated train/test error curves as function of depth */
function trainError(d){
  return Math.max(0, 0.42 * Math.exp(-0.55*d));
}
function testError(d){
  /* U-shaped: sweet spot near depth 3-4 */
  return 0.08 + 0.25*Math.exp(-0.6*(d-1)) + 0.018*(d-3)*(d-3);
}
/* After pruning, test error improves for deep trees */
function prunedTestError(d,alpha){
  var base=testError(d);
  var improvement=Math.min(base*0.6, alpha*(d-2)*0.04);
  return Math.max(0.07, base-improvement);
}

function renderOverfit(){
  var depth=S.treeDepth;
  var alpha=S.pruneAlpha;
  var trE=trainError(depth), teE=testError(depth), prE=prunedTestError(depth,alpha);
  var sweetSpot=3;

  /* error curve plot */
  var sv=plotAxes('Tree Depth','Error',8,0.5,
    [1,2,3,4,5,6,7,8],[0,0.1,0.2,0.3,0.4,0.5]);

  /* sweet spot band */
  sv+='<rect x="'+sx(2.6,8).toFixed(1)+'" y="'+PT+'" width="'+(sx(3.8,8)-sx(2.6,8)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.green,0.06)+'"/>';
  sv+='<text x="'+sx(3.2,8).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.green+'" font-size="11" font-family="monospace">sweet spot</text>';

  /* train error curve */
  var trnPath='', tstPath='', prnPath='';
  for(var d=1;d<=8;d+=0.1){
    var te=trainError(d), te2=testError(d), pe=prunedTestError(d,alpha);
    var px=sx(d,8), pyt=sy(te,0.5), pyte=sy(te2,0.5), pype=sy(pe,0.5);
    trnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyt.toFixed(1)+' ';
    tstPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyte.toFixed(1)+' ';
    prnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pype.toFixed(1)+' ';
  }
  sv+='<path d="'+trnPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
  sv+='<path d="'+tstPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';
  if(alpha>0.01) sv+='<path d="'+prnPath+'" fill="none" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';

  /* current depth vertical */
  var dvx=sx(depth,8);
  sv+='<line x1="'+dvx.toFixed(1)+'" y1="'+PT+'" x2="'+dvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(trE,0.5).toFixed(1)+'" r="5.5" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(teE,0.5).toFixed(1)+'" r="5.5" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  if(alpha>0.01) sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(prE,0.5).toFixed(1)+'" r="5.5" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-56)+'" width="150" height="54" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-44)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-44)+'" stroke="'+C.green+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-41)+'" fill="'+C.green+'" font-size="11" font-family="monospace">Train error</text>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-30)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-30)+'" stroke="'+C.red+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-27)+'" fill="'+C.red+'" font-size="11" font-family="monospace">Test error</text>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-16)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-16)+'" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-13)+'" fill="'+C.yellow+'" font-size="11" font-family="monospace">Pruned test</text>';

  var bias=depth<=2?'High':'Low';
  var variance=depth>=5?'High':'Low';
  var regime=depth<=2?'Underfitting':depth>=6?'Overfitting':'Good fit';
  var regimeCol=depth<=2?C.blue:depth>=6?C.red:C.green;

  var out=sectionTitle('Overfitting & Pruning','Depth controls the bias-variance tradeoff — constrain or prune to prevent memorisation');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Train vs Test Error by Depth')
    +svgBox(sv)
    +sliderRow('treeDepth',depth,1,8,1,'max_depth',0)
    +sliderRow('pruneAlpha',alpha,0,0.5,0.01,'ccp_alpha',2)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.blue+';">\u2190 underfitting</span>'
    +'<span style="color:'+C.red+';">overfitting \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT STATE');
  out+=statRow('max_depth',depth,C.accent);
  out+=statRow('Max leaf nodes',Math.pow(2,depth),C.purple);
  out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
  out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
  if(alpha>0.01) out+=statRow('Pruned test',(prE*100).toFixed(1)+'%',C.yellow);
  out+=statRow('Bias',bias,depth<=2?C.orange:C.green);
  out+=statRow('Variance',variance,depth>=5?C.orange:C.green);
  out+=statRow('Regime',regime,regimeCol);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRE-PRUNING PARAMS');
  [
    {p:'max_depth',     desc:'max root-to-leaf distance',    eg:'3\u20135',    c:C.accent},
    {p:'min_samples_split', desc:'min samples to split node', eg:'\u226510',   c:C.yellow},
    {p:'min_samples_leaf',  desc:'min samples in each leaf',  eg:'\u22655',    c:C.blue},
    {p:'min_impurity_decrease', desc:'only split if gain \u2265 this',eg:'0.01',c:C.purple},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
      +'<span style="color:'+r.c+';font-family:monospace;font-weight:700;">'+r.p+'</span>'
      +'<span style="color:'+C.muted+';">e.g. '+r.eg+'</span></div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','POST-PRUNING (ccp_alpha)');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Grows full tree, then removes subtrees with penalty:<br>'
    +'<span style="color:'+C.accent+';font-family:monospace;">R\u03b1(T) = R(T) + \u03b1 \u00b7 |T|</span><br>'
    +'Higher \u03b1 = simpler tree. Find optimal \u03b1 via cross-validation.<br>'
    +'sklearn: <span style="color:'+C.yellow+';font-family:monospace;">ccp_alpha</span> parameter.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127919;','The U-Curve Rule of Thumb',
    'Training error <span style="color:'+C.green+';font-weight:700;">monotonically decreases</span> with depth — a depth-n tree can perfectly memorise n training points. '
    +'Test error follows a <span style="color:'+C.red+';font-weight:700;">U-curve</span>. '
    +'Start with <span style="color:'+C.accent+';font-family:monospace;">max_depth=3</span> and increase if the model underfits. '
    +'Use <span style="color:'+C.yellow+';font-family:monospace;">cross-validation</span> to find the sweet spot — never judge by training error alone.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 4 — FEATURE IMPORTANCE (MDI)
══════════════════════════════════════════════════════ */
var FEAT_DATASETS=[
  {
    name:'Medical Diagnosis',
    features:['Blood Pressure','Cholesterol','Age','BMI','Glucose','Smoking'],
    imps:[0.38,0.25,0.18,0.10,0.06,0.03],
    cols:[C.red,C.orange,C.yellow,C.blue,C.purple,C.muted],
    rootFeat:'Blood Pressure',
    note:'BP is split at root (affects 100% of samples), giving ~2.5\u00d7 the importance of Cholesterol.'
  },
  {
    name:'Customer Churn',
    features:['Contract Type','Monthly Charges','Tenure','Tech Support','Payment Method','Gender'],
    imps:[0.42,0.27,0.15,0.09,0.05,0.02],
    cols:[C.accent,C.blue,C.green,C.yellow,C.orange,C.muted],
    rootFeat:'Contract Type',
    note:'Long-term contracts strongly predict retention. Gender has near-zero importance \u2014 as expected.'
  }
];

function renderFeatureImportance(){
  var ds=FEAT_DATASETS[S.featDataset];
  var imps=ds.imps;
  var total=imps.reduce(function(a,b){return a+b;},0);
  var norm=imps.map(function(v){return v/total;});

  /* bar chart SVG */
  var BW=500, BH=240;
  var bPL=110, bPR=20, bPT=15, bPB=20;
  var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
  var barH=Math.floor(bPH/norm.length)-6;

  var sv='';
  /* gridlines */
  [0,0.1,0.2,0.3,0.4].forEach(function(v){
    var bx=bPL+v*bPW;
    sv+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(bPT+bPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+bx.toFixed(1)+'" y="'+(bPT+bPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
  });
  sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-3)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">MDI Importance (%)</text>';

  norm.forEach(function(v,i){
    var by=bPT+i*(barH+6);
    var bw2=v*bPW;
    var col=ds.cols[i];
    /* bar bg */
    sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(col,0.08)+'"/>';
    /* bar fill */
    sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(col,0.75)+'"/>';
    /* rank badge if top 1 */
    if(i===0){
      sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3"'
         +' fill="none" stroke="'+col+'" stroke-width="1.5"/>';
      sv+='<rect x="'+(bPL+bw2+4)+'" y="'+(by+2)+'" width="30" height="'+(barH-4)+'" rx="3" fill="'+hex(col,0.2)+'" stroke="'+col+'" stroke-width="1"/>';
      sv+='<text x="'+(bPL+bw2+19)+'" y="'+(by+barH/2+3)+'" text-anchor="middle" fill="'+col+'" font-size="12" font-weight="700">ROOT</text>';
    }
    /* feature label */
    sv+='<text x="'+(bPL-6)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="'+(i<3?C.text:C.muted)+'" font-size="11" font-family="monospace">'+ds.features[i]+'</text>';
    /* value */
    sv+='<text x="'+(bPL+bw2-4)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="#0a0a0f" font-size="11" font-weight="700">'+(v*100).toFixed(1)+'%</text>';
  });

  /* MDI formula annotation */
  var formulaX=bPL+bPW*0.55, formulaY=bPT+10;
  sv+='<rect x="'+formulaX.toFixed(1)+'" y="'+formulaY.toFixed(1)+'" width="140" height="30" rx="4" fill="#0a0a0f" opacity="0.92" stroke="'+C.border+'" stroke-width="1"/>';
  sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+11).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">Importance(f) =</text>';
  sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+23).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="12" font-family="monospace">\u03a3 (|node|/n)\u00b7Gain(node)</text>';

  var out=sectionTitle('Feature Importance (MDI)','Mean Decrease in Impurity — how much each feature reduces node impurity across all splits');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  FEAT_DATASETS.forEach(function(d,i){
    out+=btnSel(i,S.featDataset,C.accent,'&#128202; '+d.name,'featData');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',ds.name+' \u2014 Feature Importances')
    +svgBox(sv,BW,BH)
    +'<div style="margin-top:10px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
    +'<div style="font-size:8.5px;color:'+C.muted+';line-height:1.7;">Root feature: <span style="color:'+ds.cols[0]+';font-weight:700;">'+ds.rootFeat+'</span> \u2014 '+ds.note+'</div>'
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  /* top features table */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TOP FEATURES');
  norm.forEach(function(v,i){
    var bar=Math.round(v*20);
    out+='<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="width:16px;text-align:right;font-size:9px;color:'+C.dim+';">'+(i+1)+'.</div>'
      +'<div style="flex:1;font-size:9px;color:'+(i<2?C.text:C.muted)+';">'+ds.features[i]+'</div>'
      +'<div style="font-size:9px;font-family:monospace;color:'+ds.cols[i]+';font-weight:700;width:36px;text-align:right;">'+(v*100).toFixed(1)+'%</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHAT MDI TELLS YOU');
  [
    {icon:'\u2714',lbl:'Features tree uses most',c:C.green},
    {icon:'\u2714',lbl:'Relative training influence',c:C.green},
    {icon:'\u2718',lbl:'Direction of relationship',c:C.red},
    {icon:'\u2718',lbl:'Causal importance',c:C.red},
    {icon:'\u26a0',lbl:'Biased toward high-cardinality features',c:C.yellow},
  ].forEach(function(r){
    out+='<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;font-size:9px;">'
      +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
      +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MDI vs PERMUTATION');
  [
    {lbl:'MDI (default)',        pro:'fast, no refit needed',   con:'biased to high cardinality', c:C.orange},
    {lbl:'Permutation',          pro:'unbiased, model-agnostic', con:'slower (n_repeats \u00d7 fit)', c:C.blue},
  ].forEach(function(r){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
      +'<div style="font-size:8px;color:'+C.red+';margin-top:1px;">\u2718 '+r.con+'</div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9888;&#65039;','MDI Bias Caveat',
    'MDI is <span style="color:'+C.red+';font-weight:700;">biased toward features with many unique values</span> — '
    +'they have more threshold candidates, increasing their chance of a high gain by chance. '
    +'For unbiased importance use <span style="color:'+C.accent+';font-family:monospace;">sklearn.inspection.permutation_importance</span> '
    +'or consider <span style="color:'+C.yellow+';font-weight:700;">SHAP values</span> for both direction and magnitude of influence.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════ */
var TABS=[
  '\u26d4 Intro &amp; Splits',
  '&#128202; Impurity Criteria',
  '&#127794; CART Growing',
  '\u2702\uFE0F Overfitting &amp; Pruning',
  '&#128161; Feature Importance'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.yellow+','+C.orange+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Decision Tree</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough \u2014 from axis-aligned splits and impurity to CART, pruning and feature importance')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderImpurity();
  else if(S.tab===2) html+=renderCART();
  else if(S.tab===3) html+=renderOverfit();
  else if(S.tab===4) html+=renderFeatureImportance();
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
        if(action==='tab')        {S.tab=idx; render();}
        else if(action==='cartStep'){S.cartStep=idx; render();}
        else if(action==='cartNext'){if(S.cartStep<4){S.cartStep++;render();}}
        else if(action==='cartPrev'){if(S.cartStep>0){S.cartStep--;render();}}
        else if(action==='featData'){S.featDataset=idx; render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='dtDepth')      {S.dtDepth=Math.round(val);  render();}
        else if(action==='pVal')    {S.pVal=val;                 render();}
        else if(action==='treeDepth'){S.treeDepth=Math.round(val);render();}
        else if(action==='pruneAlpha'){S.pruneAlpha=val;         render();}
      });
    }
  });
}

render();
/* ─── CTRL+SCROLL ZOOM ─── */
var ZOOM = 1.0;
var _zoomTimer = null;

function applyZoom(){
  document.body.style.zoom = ZOOM;
  clearTimeout(_zoomTimer);
  var toast = document.getElementById('zoom-toast');
  if(!toast){
    toast = document.createElement('div');
    toast.id = 'zoom-toast';
    toast.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#12121a;'
      +'border:1px solid #4ecdc4;color:#4ecdc4;font-family:monospace;font-size:12px;'
      +'font-weight:700;padding:8px 14px;border-radius:8px;z-index:9999;'
      +'pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(toast);
  }
  toast.textContent = 'zoom ' + Math.round(ZOOM*100) + '%';
  toast.style.opacity = '1';
  _zoomTimer = setTimeout(function(){ toast.style.opacity = '0'; }, 1200);
}

document.addEventListener('wheel', function(e){
  if(!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  ZOOM = Math.min(3.0, Math.max(0.4, ZOOM + (e.deltaY > 0 ? -0.05 : 0.05)));
  ZOOM = Math.round(ZOOM * 100) / 100;
  applyZoom();
}, {passive: false});
</script>
</body>
</html>"""

DT_VISUAL_HEIGHT = 1300

# """
# Self-contained HTML visual for Decision Trees.
# 5 interactive tabs: Intro & Axis-Aligned Splits, Impurity Criteria (Gini vs Entropy),
# CART Growing Process, Overfitting & Pruning, Feature Importance.
# Pure vanilla HTML/JS — zero CDN dependencies.
# Embed via: st.components.v1.html(DT_VISUAL_HTML, height=DT_VISUAL_HEIGHT, scrolling=True)
# """
#
# DT_VISUAL_HTML = r"""<!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <style>
# *{margin:0;padding:0;box-sizing:border-box;}
# body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
# button{cursor:pointer;font-family:inherit;}
# input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
# input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
# @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
# .fade{animation:fadeIn .3s ease both;}
# .card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
# .tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
# .tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
# .tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
# .section-title{text-align:center;margin-bottom:20px;}
# .section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
# .section-title p{font-size:11px;color:#71717a;}
# .insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
# .ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
# .ins-body{font-size:11px;color:#71717a;line-height:1.8;}
# </style>
# </head>
# <body>
# <div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
# <script>
# /* ─── PALETTE ─── */
# var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
#   accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
#   yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
#   dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
#   orange:"#fb923c"};
#
# /* ─── HELPERS ─── */
# function hex(c,a){
#   var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
#   return 'rgba('+r+','+g+','+b+','+a+')';
# }
# function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
# function card(inner,extra){
#   return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
# }
# function sectionTitle(t,s){
#   return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
# }
# function insight(icon,title,body){
#   return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
# }
# function btnSel(idx,cur,color,label,action){
#   var on=idx===cur;
#   return '<button data-action="'+action+'" data-idx="'+idx
#     +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
#     +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
# }
# function sliderRow(action,val,min,max,step,label,dec){
#   var dv=(dec!==undefined)?val.toFixed(dec):val;
#   return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
#     +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
#     +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
#     +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
#     +'</div>';
# }
# function statRow(label,val,color){
#   return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
#     +'<span style="color:'+C.muted+';">'+label+'</span>'
#     +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
# }
# function svgBox(inner,w,h){
#   return '<svg width="100%" viewBox="0 0 '+(w||520)+' '+(h||340)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
# }
#
# /* ─── SVG PLOT SCAFFOLD ─── */
# var VW=520,VH=340,PL=52,PR=18,PT=18,PB=44;
# var PW=VW-PL-PR, PH=VH-PT-PB;
# function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
# function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
# function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
#   var xm=xmax||10, ym=ymax||10;
#   var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
#   var o='';
#   xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
#   yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
#   o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
#   yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
#   o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">'+xl+'</text>';
#   o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
#   return o;
# }
#
# /* ─── STATE ─── */
# var S={
#   tab:0,
#   dtDepth:1,      /* intro split depth 1..3 */
#   pVal:0.5,       /* impurity p slider */
#   cartStep:0,     /* 0..4 growing steps */
#   treeDepth:1,    /* overfitting depth 1..8 */
#   pruneAlpha:0,   /* ccp_alpha 0..0.5 */
#   featDataset:0   /* 0=synthetic A  1=synthetic B */
# };
#
# /* ══════════════════════════════════════════════════════
#    TAB 0 — INTRO & AXIS-ALIGNED SPLITS
# ══════════════════════════════════════════════════════ */
# /* Fixed 2-class dataset with two clusters */
# var INTRO_PTS=[
#   /* class 0 (blue) — upper-left cluster */
#   {x:1.5,y:7.5,c:0},{x:2.2,y:8.8,c:0},{x:3.0,y:7.0,c:0},{x:1.8,y:9.2,c:0},
#   {x:2.8,y:6.2,c:0},{x:1.2,y:6.8,c:0},{x:3.5,y:8.5,c:0},{x:4.2,y:7.8,c:0},
#   /* class 1 (orange) — lower-right cluster */
#   {x:6.5,y:2.5,c:1},{x:7.8,y:3.8,c:1},{x:8.5,y:1.8,c:1},{x:7.2,y:1.2,c:1},
#   {x:9.0,y:3.2,c:1},{x:6.0,y:4.5,c:1},{x:8.0,y:4.8,c:1},{x:9.5,y:2.0,c:1},
#   /* overlap region */
#   {x:4.8,y:5.5,c:0},{x:5.5,y:4.8,c:1},{x:5.2,y:6.2,c:0},{x:6.2,y:5.8,c:1}
# ];
#
# /* Precomputed splits for depth 1,2,3 */
# var SPLITS=[
#   /* depth 1 */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]}],
#   /* depth 2 — add horizontal cut in right half */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
#    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]}],
#   /* depth 3 — refine left side too */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
#    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]},
#    {feat:'x\u2082',thresh:6.5,dir:'horiz',val:6.5, region:[0,5,0,10]}]
# ];
#
# function classifyPoint(p,depth){
#   /* route point through learned axis-aligned splits */
#   if(depth===0) return 0;
#   if(p.x<=5.0){
#     if(depth<3) return 0;
#     return p.y<=6.5?1:0;
#   } else {
#     if(depth<2) return 1;
#     return p.y<=5.0?1:0;
#   }
# }
#
# function regionColor(x,y,depth){
#   if(depth===0) return hex(C.blue,0.07);
#   if(x<=5.0){
#     if(depth<3) return hex(C.blue,0.07);
#     return y<=6.5?hex(C.orange,0.07):hex(C.blue,0.07);
#   } else {
#     if(depth<2) return hex(C.orange,0.07);
#     return y<=5.0?hex(C.orange,0.07):hex(C.blue,0.07);
#   }
# }
#
# function renderIntro(){
#   var depth=S.dtDepth;
#   var splits=SPLITS.slice(0,depth);
#
#   /* accuracy */
#   var correct=INTRO_PTS.filter(function(p){return classifyPoint(p,depth)===p.c;}).length;
#   var acc=(correct/INTRO_PTS.length*100).toFixed(0);
#   var accCol=acc>90?C.green:acc>75?C.yellow:C.red;
#
#   /* build SVG */
#   var sv=plotAxes('Feature x\u2081','Feature x\u2082');
#
#   /* region shading — sample grid */
#   var step=1;
#   for(var gx=0;gx<10;gx+=step){
#     for(var gy=0;gy<10;gy+=step){
#       var cx2=gx+step/2, cy2=gy+step/2;
#       sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
#          +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
#          +' fill="'+regionColor(cx2,cy2,depth)+'"/>';
#     }
#   }
#
#   /* split lines */
#   var splitColors=[C.accent,C.yellow,C.purple];
#   splits.forEach(function(sp,i){
#     var col=splitColors[i];
#     if(sp.dir==='vert'){
#       var lx=sx(sp.val);
#       sv+='<line x1="'+lx.toFixed(1)+'" y1="'+PT+'" x2="'+lx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
#       sv+='<rect x="'+(lx-52)+'" y="'+(PT+3)+'" width="100" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
#       sv+='<text x="'+lx.toFixed(1)+'" y="'+(PT+13)+'" text-anchor="middle" fill="'+col+'" font-size="11" font-family="monospace" font-weight="700">x\u2081 \u2264 '+sp.val+'?</text>';
#     } else {
#       var ly=sy(sp.val);
#       var rx1=sx(sp.region[0]), rx2=sx(sp.region[1]);
#       sv+='<line x1="'+rx1.toFixed(1)+'" y1="'+ly.toFixed(1)+'" x2="'+rx2.toFixed(1)+'" y2="'+ly.toFixed(1)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
#       sv+='<rect x="'+(rx2-110)+'" y="'+(ly-16)+'" width="108" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
#       sv+='<text x="'+(rx2-6)+'" y="'+(ly-5)+'" text-anchor="end" fill="'+col+'" font-size="11" font-family="monospace" font-weight="700">x\u2082 \u2264 '+sp.val+'?</text>';
#     }
#   });
#
#   /* data points */
#   INTRO_PTS.forEach(function(p){
#     var pred=classifyPoint(p,depth);
#     var ok=pred===p.c;
#     var col=p.c===0?C.blue:C.orange;
#     sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+col+'"'
#        +' stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'"/>';
#     if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="12">\u2717</text>';
#   });
#
#   /* accuracy badge */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-22)+'" width="120" height="18" rx="4" fill="#0a0a0f" opacity="0.9"/>';
#   sv+='<text x="'+(PL+10)+'" y="'+(PT+PH-8)+'" fill="'+accCol+'" font-size="12" font-family="monospace" font-weight="700">accuracy: '+acc+'%</text>';
#
#   var out=sectionTitle('What is a Decision Tree?','A hierarchy of axis-aligned yes/no questions that partitions feature space into regions');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   /* left: interactive plot */
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Axis-Aligned Splits')
#     +svgBox(sv)
#     +'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
#     +[{c:C.blue,l:'Class 0'},{c:C.orange,l:'Class 1'},{c:C.red,l:'Misclassified'}].map(function(i){
#       return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
#         +'<div style="width:10px;height:10px;border-radius:50%;background:'+i.c+';"></div>'+i.l+'</div>';
#     }).join('')
#     +'</div>'
#     +sliderRow('dtDepth',depth,1,3,1,'tree depth',0)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.yellow+';">\u2190 coarse (1 split)</span>'
#     +'<span style="color:'+C.green+';">fine (3 splits) \u2192</span></div>'
#   );
#   out+='</div>';
#
#   /* right: concepts */
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT TREE');
#   out+=statRow('Depth',depth,C.accent);
#   out+=statRow('Splits applied',depth,C.yellow);
#   out+=statRow('Leaf regions',depth+1,C.purple);
#   out+=statRow('Accuracy',acc+'%',accCol);
#   out+=statRow('Class 0 correct',INTRO_PTS.filter(function(p){return p.c===0&&classifyPoint(p,depth)===0;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===0;}).length,C.blue);
#   out+=statRow('Class 1 correct',INTRO_PTS.filter(function(p){return p.c===1&&classifyPoint(p,depth)===1;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===1;}).length,C.orange);
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY CONCEPTS');
#   [
#     {lbl:'No linearity assumption',c:C.green},
#     {lbl:'Axis-aligned boundaries only',c:C.yellow},
#     {lbl:'Greedy recursive splitting',c:C.blue},
#     {lbl:'Each leaf = one prediction',c:C.purple},
#     {lbl:'Depth controls complexity',c:C.orange},
#   ].forEach(function(row){
#     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<div style="width:8px;height:8px;border-radius:2px;background:'+row.c+';flex-shrink:0;"></div>'
#       +'<span style="color:'+C.muted+';">'+row.lbl+'</span></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VS LINEAR MODELS');
#   [
#     {lbl:'Linear / LR / SVM',eq:'f(x) = w\u00b7x + b',desc:'features contribute additively',c:C.blue},
#     {lbl:'Decision Tree',eq:'f(x) = lookup leaf(x)',desc:'features contribute conditionally',c:C.accent},
#   ].forEach(function(m){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:6px;border-left:3px solid '+m.c+';">'
#       +'<div style="font-size:9px;font-weight:700;color:'+m.c+';">'+m.lbl+'</div>'
#       +'<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';margin:1px 0;">'+m.eq+'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';">'+m.desc+'</div></div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#9889;','Why Trees Can Model Anything',
#     'Each additional depth level doubles the number of leaf regions. '
#     +'A tree of depth d creates up to <span style="color:'+C.yellow+';font-family:monospace;">2\u1d48</span> rectangular regions — '
#     +'together they can approximate <span style="color:'+C.accent+';font-weight:700;">any</span> decision boundary, '
#     +'no matter how non-linear. The price: depth without constraint leads to <span style="color:'+C.red+';font-weight:700;">overfitting</span>.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 1 — IMPURITY CRITERIA (GINI VS ENTROPY)
# ══════════════════════════════════════════════════════ */
# function gini(p){return 2*p*(1-p);}
# function entropy(p){
#   if(p<=0||p>=1) return 0;
#   return -p*Math.log2(p)-(1-p)*Math.log2(1-p);
# }
#
# function renderImpurity(){
#   var p=S.pVal;
#   var gv=gini(p), ev=entropy(p);
#
#   /* build impurity curve chart */
#   var sv=plotAxes('p (fraction class 1)','Impurity',1,1,
#     [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#     [0,0.2,0.4,0.6,0.8,1.0]);
#
#   /* shade under curves */
#   var gPath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
#   var ePath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
#   var steps=100;
#   for(var i=0;i<=steps;i++){
#     var pp=i/steps;
#     gPath+=' L'+sx(pp,1).toFixed(1)+','+sy(gini(pp),1).toFixed(1);
#     ePath+=' L'+sx(pp,1).toFixed(1)+','+sy(entropy(pp),1).toFixed(1);
#   }
#   var baseY=sy(0,1);
#   gPath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
#   ePath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
#   sv+='<path d="'+gPath+'" fill="'+hex(C.orange,0.12)+'"/>';
#   sv+='<path d="'+ePath+'" fill="'+hex(C.blue,0.10)+'"/>';
#
#   /* curve lines */
#   var gLine='', eLine='';
#   for(var j=0;j<=steps;j++){
#     var pj=j/steps;
#     var gpt=sx(pj,1).toFixed(1)+','+sy(gini(pj),1).toFixed(1);
#     var ept=sx(pj,1).toFixed(1)+','+sy(entropy(pj),1).toFixed(1);
#     gLine+=(j===0?'M':'L')+gpt+' ';
#     eLine+=(j===0?'M':'L')+ept+' ';
#   }
#   sv+='<path d="'+gLine+'" fill="none" stroke="'+C.orange+'" stroke-width="2.2"/>';
#   sv+='<path d="'+eLine+'" fill="none" stroke="'+C.blue+'" stroke-width="2.2"/>';
#
#   /* vertical cursor */
#   var cx3=sx(p,1);
#   sv+='<line x1="'+cx3.toFixed(1)+'" y1="'+PT+'" x2="'+cx3.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
#   /* dots on curves */
#   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(gv,1).toFixed(1)+'" r="6.5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(ev,1).toFixed(1)+'" r="6.5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#
#   /* value labels */
#   var lx=cx3+8;
#   sv+='<rect x="'+(lx-2)+'" y="'+(sy(gv,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
#   sv+='<text x="'+lx+'" y="'+(sy(gv,1)+0)+'" fill="'+C.orange+'" font-size="11" font-family="monospace">Gini='+gv.toFixed(3)+'</text>';
#   sv+='<rect x="'+(lx-2)+'" y="'+(sy(ev,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
#   sv+='<text x="'+lx+'" y="'+(sy(ev,1)+0)+'" fill="'+C.blue+'" font-size="11" font-family="monospace">H='+ev.toFixed(3)+'</text>';
#
#   /* p label */
#   sv+='<text x="'+cx3.toFixed(1)+'" y="'+(PT+PH+26)+'" text-anchor="middle" fill="'+C.accent+'" font-size="11" font-weight="700">p='+p.toFixed(2)+'</text>';
#
#   /* legend */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="138" height="36" rx="4" fill="#0a0a0f" opacity="0.88"/>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+13)+'" x2="'+(PL+30)+'" y2="'+(PT+13)+'" stroke="'+C.orange+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+16)+'" fill="'+C.orange+'" font-size="11" font-family="monospace">Gini = 2p(1\u2212p)</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+28)+'" x2="'+(PL+30)+'" y2="'+(PT+28)+'" stroke="'+C.blue+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+31)+'" fill="'+C.blue+'" font-size="11" font-family="monospace">H = \u2212p\u00b7log\u2082p\u2026</text>';
#
#   /* ─── Gain demo: candidate split ─── */
#   var nParent=10, nPos=Math.round(p*nParent), nNeg=nParent-nPos;
#   var gParent=gini(p), hParent=entropy(p);
#   /* split: left 5/10, right 5/10 */
#   var nL=5, pL=Math.min(1,Math.max(0,Math.round(p*1.2*nL)/nL));
#   var nR=5, pR=Math.min(1,Math.max(0,Math.round(p*0.8*nR)/nR));
#   pL=Math.min(1,pL); pR=Math.min(1,pR);
#   var wGini=(nL/nParent)*gini(pL)+(nR/nParent)*gini(pR);
#   var wH   =(nL/nParent)*entropy(pL)+(nR/nParent)*entropy(pR);
#   var gGain=Math.max(0,gParent-wGini);
#   var igGain=Math.max(0,hParent-wH);
#   var gainCol=gGain>0.1?C.green:gGain>0.02?C.yellow:C.red;
#
#   var out=sectionTitle('Impurity Criteria','Gini and Entropy measure how mixed the class labels are at each node');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Impurity vs Class Proportion p')
#     +svgBox(sv)
#     +sliderRow('pVal',p,0.01,0.99,0.01,'p (class 1)',2)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.green+';">\u2190 pure class 0 (impurity = 0)</span>'
#     +'<span style="color:'+C.green+';">pure class 1 \u2192</span></div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#
#   /* current impurity values */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT p = '+p.toFixed(2));
#   out+=statRow('Gini impurity',''+gv.toFixed(4),C.orange);
#   out+=statRow('Entropy H',''+ev.toFixed(4)+' bits',C.blue);
#   out+=statRow('Is pure?',(p<=0.01||p>=0.99)?'Yes \u2713':'No',(p<=0.01||p>=0.99)?C.green:C.red);
#   out+=statRow('Max impurity',(Math.abs(p-0.5)<0.05)?'Yes \u2248 0.5':'No',(Math.abs(p-0.5)<0.05)?C.yellow:C.muted);
#   out+='</div>';
#
#   /* formulas */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','FORMULAS');
#   [
#     {lbl:'Gini (binary)',    eq:'1 \u2212 p\u00b2 \u2212 (1\u2212p)\u00b2',  sub:'= 2p(1\u2212p)',  c:C.orange},
#     {lbl:'Entropy (binary)', eq:'\u2212p\u00b7log\u2082p \u2212 (1\u2212p)\u00b7log\u2082(1\u2212p)', sub:'max at p=0.5: H=1 bit',c:C.blue},
#     {lbl:'Gini general',     eq:'1 \u2212 \u2211\u2c7c p\u2c7c\u00b2',          sub:'k classes',         c:C.yellow},
#   ].forEach(function(f){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+f.c+';">'
#       +'<div style="font-size:8.5px;color:'+C.dim+';">'+f.lbl+'</div>'
#       +'<div style="font-size:9.5px;font-family:monospace;color:'+f.c+';margin:2px 0;">'+f.eq+'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';">'+f.sub+'</div></div>';
#   });
#   out+='</div>';
#
#   /* information gain demo */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INFORMATION GAIN DEMO');
#   out+='<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Parent: '+nPos+' pos / '+nNeg+' neg (p='+p.toFixed(2)+')</div>';
#   [
#     {lbl:'Gini(parent)',   val:gParent.toFixed(4), c:C.orange},
#     {lbl:'Weighted Gini',  val:wGini.toFixed(4),   c:C.orange},
#     {lbl:'Gini Gain',      val:gGain.toFixed(4),   c:gainCol},
#     {lbl:'H(parent)',      val:hParent.toFixed(4)+' bits',c:C.blue},
#     {lbl:'Info Gain (IG)', val:igGain.toFixed(4)+' bits', c:gainCol},
#   ].forEach(function(r){out+=statRow(r.lbl,r.val,r.c);});
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#128200;','Gini vs Entropy in Practice',
#     'Gini is <span style="color:'+C.green+';font-weight:700;">slightly faster</span> (no log) and is the sklearn default. '
#     +'Entropy is <span style="color:'+C.yellow+';font-weight:700;">more sensitive near pure nodes</span> and can prefer different splits on ambiguous data. '
#     +'In practice, <span style="color:'+C.accent+';font-weight:700;">both produce nearly identical trees</span> — the choice rarely changes final accuracy by more than ~1%.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 2 — CART GROWING PROCESS
# ══════════════════════════════════════════════════════ */
# var CART_STEPS=[
#   {
#     title:'Step 0 — All data at root',
#     desc:'All 10 training examples land at the root. We compute the parent Gini impurity (6 Yes, 4 No \u2192 p=0.6).',
#     detail:'Gini(parent) = 1 \u2212 0.6\u00b2 \u2212 0.4\u00b2 = 1 \u2212 0.36 \u2212 0.16 = 0.48',
#     splitLabel:null,
#     nodes:[
#       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true}
#     ],
#     edges:[]
#   },
#   {
#     title:'Step 1 — Evaluate candidate splits',
#     desc:'For each feature and threshold, compute weighted child Gini. Find the split with maximum Gain.',
#     detail:'Age \u2264 42: Gain = 0.02 (small)\nIncome = High: Gain = 0.48 \u2714 (maximum!)',
#     splitLabel:'Best split: Income = High',
#     nodes:[
#       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true},
#       {id:'cA',x:100,y:128,w:130,h:40,label:'Income = High',sub:'Gini = 0.48 (Age \u226442)',col:C.red,active:false,dim:true},
#       {id:'cB',x:280,y:128,w:130,h:40,label:'Income = High',sub:'Gain = 0.48 \u2713 BEST',col:C.green,active:true},
#     ],
#     edges:[
#       {from:'root',to:'cA',label:'Age\u226442',col:C.dim},
#       {from:'root',to:'cB',label:'Income',col:C.green}
#     ]
#   },
#   {
#     title:'Step 2 — Apply best split',
#     desc:'Split on Income = High. Left child (High): 6 Yes, 0 No — PURE! Right child (Low): 0 Yes, 4 No — PURE!',
#     detail:'Weighted Gini = (6/10)\u00b70 + (4/10)\u00b70 = 0.00  \u2192  Gini Gain = 0.48',
#     splitLabel:'Both children are pure leaves \u2713',
#     nodes:[
#       {id:'root',x:220,y:22,w:170,h:40,label:'Income = High?',sub:'Gain=0.48 | n=10',col:C.accent,active:false},
#       {id:'left',x:100,y:120,w:140,h:46,label:'\u2714 High (Yes)',sub:'6 Yes, 0 No\nGini = 0.00 (PURE)',col:C.green,active:true},
#       {id:'right',x:300,y:120,w:140,h:46,label:'\u2718 Low (No)',sub:'0 Yes, 4 No\nGini = 0.00 (PURE)',col:C.green,active:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'Yes (High)',col:C.green},
#       {from:'root',to:'right',label:'No (Low)', col:C.orange}
#     ]
#   },
#   {
#     title:'Step 3 — Stopping conditions checked',
#     desc:'Both children are pure (Gini = 0). Recursion stops. Leaves are assigned majority class predictions.',
#     detail:'Stopping rule: node is pure \u2192 create leaf\nOther rules: max_depth reached, min_samples_split, zero gain',
#     splitLabel:'Tree fully grown \u2014 depth = 1',
#     nodes:[
#       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'root node | n=10',col:C.accent,active:false},
#       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES',sub:'Leaf | Gini=0 | n=6',col:C.green,active:true,leaf:true},
#       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | Gini=0 | n=4',col:C.blue,active:true,leaf:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'High',col:C.green},
#       {from:'root',to:'right',label:'Low', col:C.orange}
#     ]
#   },
#   {
#     title:'Step 4 — Making a prediction',
#     desc:'New example: Age=35, Income=High. Route from root \u2192 follow "High" branch \u2192 reach leaf \u2192 Predict: YES.',
#     detail:'Runtime: O(depth) = O(1) for this tree.\nOnly the path root\u2192leaf is evaluated \u2014 all other nodes are ignored.',
#     splitLabel:'Prediction in O(depth) time',
#     nodes:[
#       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'New example: Income=High',col:C.yellow,active:true},
#       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES \u2714',sub:'New example lands here',col:C.green,active:true,leaf:true,highlight:true},
#       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | n=4',col:C.blue,active:false,leaf:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'\u2714 High \u2192 here',col:C.yellow},
#       {from:'root',to:'right',label:'Low',col:C.dim}
#     ]
#   }
# ];
#
# function treeNode(n){
#   var fill=n.dim?hex(C.card,0.5):(n.highlight?hex(n.col,0.25):hex(n.col,0.12));
#   var border=n.dim?C.dim:(n.active?n.col:C.border);
#   var bw=n.active?2:1;
#   var lines=n.sub.split('\n');
#   return '<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'"'
#     +' rx="7" fill="'+fill+'" stroke="'+border+'" stroke-width="'+bw+'"/>'
#     +(n.leaf?'<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="4" rx="2" fill="'+border+'"/>':'')
#     +'<text x="'+n.x+'" y="'+(n.y+14)+'" text-anchor="middle" fill="'+(n.dim?C.dim:n.col)+'" font-size="12" font-family="monospace" font-weight="700">'+n.label+'</text>'
#     +lines.map(function(l,i){
#       return '<text x="'+n.x+'" y="'+(n.y+27+i*11)+'" text-anchor="middle" fill="'+(n.dim?C.dim:C.muted)+'" font-size="12" font-family="monospace">'+l+'</text>';
#     }).join('');
# }
#
# function treeEdge(e,nodes){
#   var fromNode=nodes.find(function(n){return n.id===e.from;});
#   var toNode=nodes.find(function(n){return n.id===e.to;});
#   if(!fromNode||!toNode) return '';
#   var x1=fromNode.x, y1=fromNode.y+fromNode.h;
#   var x2=toNode.x, y2=toNode.y;
#   var mx=(x1+x2)/2, my=(y1+y2)/2;
#   return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+e.col+'" stroke-width="1.5"/>'
#     +'<rect x="'+(mx-28)+'" y="'+(my-8)+'" width="56" height="12" rx="3" fill="#0a0a0f" opacity="0.85"/>'
#     +'<text x="'+mx+'" y="'+(my+2)+'" text-anchor="middle" fill="'+e.col+'" font-size="11" font-family="monospace">'+e.label+'</text>';
# }
#
# function renderCART(){
#   var step=S.cartStep;
#   var cs=CART_STEPS[step];
#
#   var svInner='';
#   cs.edges.forEach(function(e){ svInner+=treeEdge(e,cs.nodes); });
#   cs.nodes.forEach(function(n){ svInner+=treeNode(n); });
#
#   var out=sectionTitle('CART Growing Process','The greedy recursive algorithm that builds decision trees one split at a time');
#
#   /* step navigation */
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
#   CART_STEPS.forEach(function(st,i){
#     var on=i===step;
#     out+='<button data-action="cartStep" data-idx="'+i+'"'
#       +' style="padding:7px 12px;border-radius:8px;font-size:9.5px;font-weight:700;font-family:inherit;'
#       +'background:'+(on?hex(C.accent,.15):C.card)+';border:1.5px solid '+(on?C.accent:C.border)+';'
#       +'color:'+(on?C.accent:C.muted)+';cursor:pointer;transition:all .2s;">'
#       +'Step '+i+'</button>';
#   });
#   out+='</div>';
#
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   /* tree diagram */
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',cs.title)
#     +svgBox(svInner,520,240)
#   );
#   out+='</div>';
#
#   /* step explanation */
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:10px;','WHAT\'S HAPPENING');
#   out+=div('font-size:9.5px;color:'+C.text+';line-height:1.8;margin-bottom:10px;',cs.desc);
#   out+='<div style="background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';">'
#     +'<pre style="font-size:8.5px;color:'+C.muted+';white-space:pre-wrap;line-height:1.7;font-family:monospace;">'+cs.detail+'</pre>'
#     +'</div>';
#   if(cs.splitLabel){
#     out+='<div style="margin-top:10px;padding:8px 12px;background:'+hex(C.green,.08)+';border-radius:6px;border:1px solid '+hex(C.green,.3)+';font-size:9.5px;font-weight:700;color:'+C.green+';">'
#       +'\u2714\ufe0f '+cs.splitLabel+'</div>';
#   }
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','STOPPING CONDITIONS');
#   [
#     {c:C.green,  lbl:'Node is pure (Gini=0)', active:step>=2},
#     {c:C.yellow, lbl:'max_depth reached',     active:false},
#     {c:C.blue,   lbl:'min_samples_split',     active:false},
#     {c:C.orange, lbl:'Zero gain split',       active:false},
#   ].forEach(function(r){
#     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<div style="width:8px;height:8px;border-radius:50%;background:'+(r.active?r.c:C.dim)+';flex-shrink:0;"></div>'
#       +'<span style="color:'+(r.active?r.c:C.dim)+';">'+r.lbl+(r.active?' \u2714':'')+'</span></div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   /* navigation buttons */
#   out+='<div style="display:flex;gap:10px;justify-content:center;margin-bottom:12px;">';
#   out+='<button data-action="cartPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(step>0?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
#     +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Prev</button>';
#   out+='<button data-action="cartNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(step<4?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
#     +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next \u2192</button>';
#   out+='</div>';
#
#   out+=insight('&#128296;','Why Greedy? Why Not Global?',
#     'Finding the <span style="color:'+C.red+';font-weight:700;">globally optimal tree</span> is NP-hard — '
#     +'the number of possible trees grows super-exponentially with depth and features. '
#     +'CART\'s <span style="color:'+C.accent+';font-weight:700;">greedy approach</span> makes locally optimal decisions at each node in O(np log n) time. '
#     +'The greedy solution is good enough in practice — '
#     +'and ensembles like <span style="color:'+C.yellow+';font-weight:700;">Random Forests</span> compensate for its suboptimality.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 3 — OVERFITTING & PRUNING
# ══════════════════════════════════════════════════════ */
# /* Simulated train/test error curves as function of depth */
# function trainError(d){
#   return Math.max(0, 0.42 * Math.exp(-0.55*d));
# }
# function testError(d){
#   /* U-shaped: sweet spot near depth 3-4 */
#   return 0.08 + 0.25*Math.exp(-0.6*(d-1)) + 0.018*(d-3)*(d-3);
# }
# /* After pruning, test error improves for deep trees */
# function prunedTestError(d,alpha){
#   var base=testError(d);
#   var improvement=Math.min(base*0.6, alpha*(d-2)*0.04);
#   return Math.max(0.07, base-improvement);
# }
#
# function renderOverfit(){
#   var depth=S.treeDepth;
#   var alpha=S.pruneAlpha;
#   var trE=trainError(depth), teE=testError(depth), prE=prunedTestError(depth,alpha);
#   var sweetSpot=3;
#
#   /* error curve plot */
#   var sv=plotAxes('Tree Depth','Error',8,0.5,
#     [1,2,3,4,5,6,7,8],[0,0.1,0.2,0.3,0.4,0.5]);
#
#   /* sweet spot band */
#   sv+='<rect x="'+sx(2.6,8).toFixed(1)+'" y="'+PT+'" width="'+(sx(3.8,8)-sx(2.6,8)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.green,0.06)+'"/>';
#   sv+='<text x="'+sx(3.2,8).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.green+'" font-size="11" font-family="monospace">sweet spot</text>';
#
#   /* train error curve */
#   var trnPath='', tstPath='', prnPath='';
#   for(var d=1;d<=8;d+=0.1){
#     var te=trainError(d), te2=testError(d), pe=prunedTestError(d,alpha);
#     var px=sx(d,8), pyt=sy(te,0.5), pyte=sy(te2,0.5), pype=sy(pe,0.5);
#     trnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyt.toFixed(1)+' ';
#     tstPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyte.toFixed(1)+' ';
#     prnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pype.toFixed(1)+' ';
#   }
#   sv+='<path d="'+trnPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
#   sv+='<path d="'+tstPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';
#   if(alpha>0.01) sv+='<path d="'+prnPath+'" fill="none" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
#
#   /* current depth vertical */
#   var dvx=sx(depth,8);
#   sv+='<line x1="'+dvx.toFixed(1)+'" y1="'+PT+'" x2="'+dvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
#   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(trE,0.5).toFixed(1)+'" r="5.5" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(teE,0.5).toFixed(1)+'" r="5.5" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   if(alpha>0.01) sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(prE,0.5).toFixed(1)+'" r="5.5" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#
#   /* legend */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-56)+'" width="150" height="54" rx="4" fill="#0a0a0f" opacity="0.9"/>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-44)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-44)+'" stroke="'+C.green+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-41)+'" fill="'+C.green+'" font-size="11" font-family="monospace">Train error</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-30)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-30)+'" stroke="'+C.red+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-27)+'" fill="'+C.red+'" font-size="11" font-family="monospace">Test error</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-16)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-16)+'" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-13)+'" fill="'+C.yellow+'" font-size="11" font-family="monospace">Pruned test</text>';
#
#   var bias=depth<=2?'High':'Low';
#   var variance=depth>=5?'High':'Low';
#   var regime=depth<=2?'Underfitting':depth>=6?'Overfitting':'Good fit';
#   var regimeCol=depth<=2?C.blue:depth>=6?C.red:C.green;
#
#   var out=sectionTitle('Overfitting & Pruning','Depth controls the bias-variance tradeoff — constrain or prune to prevent memorisation');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Train vs Test Error by Depth')
#     +svgBox(sv)
#     +sliderRow('treeDepth',depth,1,8,1,'max_depth',0)
#     +sliderRow('pruneAlpha',alpha,0,0.5,0.01,'ccp_alpha',2)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.blue+';">\u2190 underfitting</span>'
#     +'<span style="color:'+C.red+';">overfitting \u2192</span></div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT STATE');
#   out+=statRow('max_depth',depth,C.accent);
#   out+=statRow('Max leaf nodes',Math.pow(2,depth),C.purple);
#   out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
#   out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
#   if(alpha>0.01) out+=statRow('Pruned test',(prE*100).toFixed(1)+'%',C.yellow);
#   out+=statRow('Bias',bias,depth<=2?C.orange:C.green);
#   out+=statRow('Variance',variance,depth>=5?C.orange:C.green);
#   out+=statRow('Regime',regime,regimeCol);
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRE-PRUNING PARAMS');
#   [
#     {p:'max_depth',     desc:'max root-to-leaf distance',    eg:'3\u20135',    c:C.accent},
#     {p:'min_samples_split', desc:'min samples to split node', eg:'\u226510',   c:C.yellow},
#     {p:'min_samples_leaf',  desc:'min samples in each leaf',  eg:'\u22655',    c:C.blue},
#     {p:'min_impurity_decrease', desc:'only split if gain \u2265 this',eg:'0.01',c:C.purple},
#   ].forEach(function(r){
#     out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
#       +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
#       +'<span style="color:'+r.c+';font-family:monospace;font-weight:700;">'+r.p+'</span>'
#       +'<span style="color:'+C.muted+';">e.g. '+r.eg+'</span></div>'
#       +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','POST-PRUNING (ccp_alpha)');
#   out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
#     'Grows full tree, then removes subtrees with penalty:<br>'
#     +'<span style="color:'+C.accent+';font-family:monospace;">R\u03b1(T) = R(T) + \u03b1 \u00b7 |T|</span><br>'
#     +'Higher \u03b1 = simpler tree. Find optimal \u03b1 via cross-validation.<br>'
#     +'sklearn: <span style="color:'+C.yellow+';font-family:monospace;">ccp_alpha</span> parameter.'
#   );
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#127919;','The U-Curve Rule of Thumb',
#     'Training error <span style="color:'+C.green+';font-weight:700;">monotonically decreases</span> with depth — a depth-n tree can perfectly memorise n training points. '
#     +'Test error follows a <span style="color:'+C.red+';font-weight:700;">U-curve</span>. '
#     +'Start with <span style="color:'+C.accent+';font-family:monospace;">max_depth=3</span> and increase if the model underfits. '
#     +'Use <span style="color:'+C.yellow+';font-family:monospace;">cross-validation</span> to find the sweet spot — never judge by training error alone.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 4 — FEATURE IMPORTANCE (MDI)
# ══════════════════════════════════════════════════════ */
# var FEAT_DATASETS=[
#   {
#     name:'Medical Diagnosis',
#     features:['Blood Pressure','Cholesterol','Age','BMI','Glucose','Smoking'],
#     imps:[0.38,0.25,0.18,0.10,0.06,0.03],
#     cols:[C.red,C.orange,C.yellow,C.blue,C.purple,C.muted],
#     rootFeat:'Blood Pressure',
#     note:'BP is split at root (affects 100% of samples), giving ~2.5\u00d7 the importance of Cholesterol.'
#   },
#   {
#     name:'Customer Churn',
#     features:['Contract Type','Monthly Charges','Tenure','Tech Support','Payment Method','Gender'],
#     imps:[0.42,0.27,0.15,0.09,0.05,0.02],
#     cols:[C.accent,C.blue,C.green,C.yellow,C.orange,C.muted],
#     rootFeat:'Contract Type',
#     note:'Long-term contracts strongly predict retention. Gender has near-zero importance \u2014 as expected.'
#   }
# ];
#
# function renderFeatureImportance(){
#   var ds=FEAT_DATASETS[S.featDataset];
#   var imps=ds.imps;
#   var total=imps.reduce(function(a,b){return a+b;},0);
#   var norm=imps.map(function(v){return v/total;});
#
#   /* bar chart SVG */
#   var BW=500, BH=240;
#   var bPL=110, bPR=20, bPT=15, bPB=20;
#   var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
#   var barH=Math.floor(bPH/norm.length)-6;
#
#   var sv='';
#   /* gridlines */
#   [0,0.1,0.2,0.3,0.4].forEach(function(v){
#     var bx=bPL+v*bPW;
#     sv+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(bPT+bPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
#     sv+='<text x="'+bx.toFixed(1)+'" y="'+(bPT+bPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
#   });
#   sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   sv+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-3)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">MDI Importance (%)</text>';
#
#   norm.forEach(function(v,i){
#     var by=bPT+i*(barH+6);
#     var bw2=v*bPW;
#     var col=ds.cols[i];
#     /* bar bg */
#     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(col,0.08)+'"/>';
#     /* bar fill */
#     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(col,0.75)+'"/>';
#     /* rank badge if top 1 */
#     if(i===0){
#       sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3"'
#          +' fill="none" stroke="'+col+'" stroke-width="1.5"/>';
#       sv+='<rect x="'+(bPL+bw2+4)+'" y="'+(by+2)+'" width="30" height="'+(barH-4)+'" rx="3" fill="'+hex(col,0.2)+'" stroke="'+col+'" stroke-width="1"/>';
#       sv+='<text x="'+(bPL+bw2+19)+'" y="'+(by+barH/2+3)+'" text-anchor="middle" fill="'+col+'" font-size="12" font-weight="700">ROOT</text>';
#     }
#     /* feature label */
#     sv+='<text x="'+(bPL-6)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="'+(i<3?C.text:C.muted)+'" font-size="11" font-family="monospace">'+ds.features[i]+'</text>';
#     /* value */
#     sv+='<text x="'+(bPL+bw2-4)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="#0a0a0f" font-size="11" font-weight="700">'+(v*100).toFixed(1)+'%</text>';
#   });
#
#   /* MDI formula annotation */
#   var formulaX=bPL+bPW*0.55, formulaY=bPT+10;
#   sv+='<rect x="'+formulaX.toFixed(1)+'" y="'+formulaY.toFixed(1)+'" width="140" height="30" rx="4" fill="#0a0a0f" opacity="0.92" stroke="'+C.border+'" stroke-width="1"/>';
#   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+11).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">Importance(f) =</text>';
#   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+23).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="12" font-family="monospace">\u03a3 (|node|/n)\u00b7Gain(node)</text>';
#
#   var out=sectionTitle('Feature Importance (MDI)','Mean Decrease in Impurity — how much each feature reduces node impurity across all splits');
#
#   out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
#   FEAT_DATASETS.forEach(function(d,i){
#     out+=btnSel(i,S.featDataset,C.accent,'\u{1F4CA} '+d.name,'featData');
#   });
#   out+='</div>';
#
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',ds.name+' \u2014 Feature Importances')
#     +svgBox(sv,BW,BH)
#     +'<div style="margin-top:10px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
#     +'<div style="font-size:8.5px;color:'+C.muted+';line-height:1.7;">Root feature: <span style="color:'+ds.cols[0]+';font-weight:700;">'+ds.rootFeat+'</span> \u2014 '+ds.note+'</div>'
#     +'</div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#
#   /* top features table */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TOP FEATURES');
#   norm.forEach(function(v,i){
#     var bar=Math.round(v*20);
#     out+='<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
#       +'<div style="width:16px;text-align:right;font-size:9px;color:'+C.dim+';">'+(i+1)+'.</div>'
#       +'<div style="flex:1;font-size:9px;color:'+(i<2?C.text:C.muted)+';">'+ds.features[i]+'</div>'
#       +'<div style="font-size:9px;font-family:monospace;color:'+ds.cols[i]+';font-weight:700;width:36px;text-align:right;">'+(v*100).toFixed(1)+'%</div>'
#       +'</div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHAT MDI TELLS YOU');
#   [
#     {icon:'\u2714',lbl:'Features tree uses most',c:C.green},
#     {icon:'\u2714',lbl:'Relative training influence',c:C.green},
#     {icon:'\u2718',lbl:'Direction of relationship',c:C.red},
#     {icon:'\u2718',lbl:'Causal importance',c:C.red},
#     {icon:'\u26a0',lbl:'Biased toward high-cardinality features',c:C.yellow},
#   ].forEach(function(r){
#     out+='<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
#       +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MDI vs PERMUTATION');
#   [
#     {lbl:'MDI (default)',        pro:'fast, no refit needed',   con:'biased to high cardinality', c:C.orange},
#     {lbl:'Permutation',          pro:'unbiased, model-agnostic', con:'slower (n_repeats \u00d7 fit)', c:C.blue},
#   ].forEach(function(r){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
#       +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
#       +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
#       +'<div style="font-size:8px;color:'+C.red+';margin-top:1px;">\u2718 '+r.con+'</div>'
#       +'</div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#9888;&#65039;','MDI Bias Caveat',
#     'MDI is <span style="color:'+C.red+';font-weight:700;">biased toward features with many unique values</span> — '
#     +'they have more threshold candidates, increasing their chance of a high gain by chance. '
#     +'For unbiased importance use <span style="color:'+C.accent+';font-family:monospace;">sklearn.inspection.permutation_importance</span> '
#     +'or consider <span style="color:'+C.yellow+';font-weight:700;">SHAP values</span> for both direction and magnitude of influence.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    ROOT RENDER
# ══════════════════════════════════════════════════════ */
# var TABS=[
#   '\u26d4 Intro &amp; Splits',
#   '\u{1F4CA} Impurity Criteria',
#   '\u{1F332} CART Growing',
#   '\u2702\uFE0F Overfitting &amp; Pruning',
#   '\u{1F4A1} Feature Importance'
# ];
#
# function renderApp(){
#   var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
#   html+='<div style="text-align:center;margin-bottom:16px;">'
#     +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.yellow+','+C.orange+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Decision Tree</div>'
#     +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough \u2014 from axis-aligned splits and impurity to CART, pruning and feature importance')
#     +'</div>';
#   html+='<div class="tab-bar">';
#   TABS.forEach(function(t,i){
#     html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
#   });
#   html+='</div>';
#   html+='<div class="fade">';
#   if(S.tab===0)      html+=renderIntro();
#   else if(S.tab===1) html+=renderImpurity();
#   else if(S.tab===2) html+=renderCART();
#   else if(S.tab===3) html+=renderOverfit();
#   else if(S.tab===4) html+=renderFeatureImportance();
#   html+='</div></div>';
#   return html;
# }
#
# function render(){
#   document.getElementById('app').innerHTML=renderApp();
#   bindEvents();
# }
#
# function bindEvents(){
#   document.querySelectorAll('[data-action]').forEach(function(el){
#     var action=el.getAttribute('data-action');
#     var idx=parseInt(el.getAttribute('data-idx'));
#     var tag=el.tagName.toLowerCase();
#     if(tag==='button'){
#       el.addEventListener('click',function(){
#         if(action==='tab')        {S.tab=idx; render();}
#         else if(action==='cartStep'){S.cartStep=idx; render();}
#         else if(action==='cartNext'){if(S.cartStep<4){S.cartStep++;render();}}
#         else if(action==='cartPrev'){if(S.cartStep>0){S.cartStep--;render();}}
#         else if(action==='featData'){S.featDataset=idx; render();}
#       });
#     } else if(tag==='input'){
#       el.addEventListener('input',function(){
#         var val=parseFloat(this.value);
#         if(action==='dtDepth')      {S.dtDepth=Math.round(val);  render();}
#         else if(action==='pVal')    {S.pVal=val;                 render();}
#         else if(action==='treeDepth'){S.treeDepth=Math.round(val);render();}
#         else if(action==='pruneAlpha'){S.pruneAlpha=val;         render();}
#       });
#     }
#   });
# }
#
# render();
# </script>
# </body>
# </html>"""
#
# DT_VISUAL_HEIGHT = 1300

# """
# Self-contained HTML visual for Decision Trees.
# 5 interactive tabs: Intro & Axis-Aligned Splits, Impurity Criteria (Gini vs Entropy),
# CART Growing Process, Overfitting & Pruning, Feature Importance.
# Pure vanilla HTML/JS — zero CDN dependencies.
# Embed via: st.components.v1.html(DT_VISUAL_HTML, height=DT_VISUAL_HEIGHT, scrolling=True)
# """
#
# DT_VISUAL_HTML = r"""<!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <style>
# *{margin:0;padding:0;box-sizing:border-box;}
# body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
# button{cursor:pointer;font-family:inherit;}
# input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
# input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
# @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
# .fade{animation:fadeIn .3s ease both;}
# .card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
# .tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
# .tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
# .tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
# .section-title{text-align:center;margin-bottom:20px;}
# .section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
# .section-title p{font-size:11px;color:#71717a;}
# .insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
# .ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
# .ins-body{font-size:11px;color:#71717a;line-height:1.8;}
# </style>
# </head>
# <body>
# <div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
# <script>
# /* ─── PALETTE ─── */
# var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
#   accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
#   yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
#   dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
#   orange:"#fb923c"};
#
# /* ─── HELPERS ─── */
# function hex(c,a){
#   var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
#   return 'rgba('+r+','+g+','+b+','+a+')';
# }
# function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
# function card(inner,extra){
#   return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
# }
# function sectionTitle(t,s){
#   return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
# }
# function insight(icon,title,body){
#   return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
# }
# function btnSel(idx,cur,color,label,action){
#   var on=idx===cur;
#   return '<button data-action="'+action+'" data-idx="'+idx
#     +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
#     +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
# }
# function sliderRow(action,val,min,max,step,label,dec){
#   var dv=(dec!==undefined)?val.toFixed(dec):val;
#   return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
#     +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
#     +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
#     +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
#     +'</div>';
# }
# function statRow(label,val,color){
#   return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
#     +'<span style="color:'+C.muted+';">'+label+'</span>'
#     +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
# }
# function svgBox(inner,w,h){
#   return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
# }
#
# /* ─── SVG PLOT SCAFFOLD ─── */
# var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
# var PW=VW-PL-PR, PH=VH-PT-PB;
# function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
# function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
# function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
#   var xm=xmax||10, ym=ymax||10;
#   var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
#   var o='';
#   xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
#   yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
#   o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
#   yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
#   o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
#   o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
#   return o;
# }
#
# /* ─── STATE ─── */
# var S={
#   tab:0,
#   dtDepth:1,      /* intro split depth 1..3 */
#   pVal:0.5,       /* impurity p slider */
#   cartStep:0,     /* 0..4 growing steps */
#   treeDepth:1,    /* overfitting depth 1..8 */
#   pruneAlpha:0,   /* ccp_alpha 0..0.5 */
#   featDataset:0   /* 0=synthetic A  1=synthetic B */
# };
#
# /* ══════════════════════════════════════════════════════
#    TAB 0 — INTRO & AXIS-ALIGNED SPLITS
# ══════════════════════════════════════════════════════ */
# /* Fixed 2-class dataset with two clusters */
# var INTRO_PTS=[
#   /* class 0 (blue) — upper-left cluster */
#   {x:1.5,y:7.5,c:0},{x:2.2,y:8.8,c:0},{x:3.0,y:7.0,c:0},{x:1.8,y:9.2,c:0},
#   {x:2.8,y:6.2,c:0},{x:1.2,y:6.8,c:0},{x:3.5,y:8.5,c:0},{x:4.2,y:7.8,c:0},
#   /* class 1 (orange) — lower-right cluster */
#   {x:6.5,y:2.5,c:1},{x:7.8,y:3.8,c:1},{x:8.5,y:1.8,c:1},{x:7.2,y:1.2,c:1},
#   {x:9.0,y:3.2,c:1},{x:6.0,y:4.5,c:1},{x:8.0,y:4.8,c:1},{x:9.5,y:2.0,c:1},
#   /* overlap region */
#   {x:4.8,y:5.5,c:0},{x:5.5,y:4.8,c:1},{x:5.2,y:6.2,c:0},{x:6.2,y:5.8,c:1}
# ];
#
# /* Precomputed splits for depth 1,2,3 */
# var SPLITS=[
#   /* depth 1 */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]}],
#   /* depth 2 — add horizontal cut in right half */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
#    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]}],
#   /* depth 3 — refine left side too */
#   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
#    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]},
#    {feat:'x\u2082',thresh:6.5,dir:'horiz',val:6.5, region:[0,5,0,10]}]
# ];
#
# function classifyPoint(p,depth){
#   /* route point through learned axis-aligned splits */
#   if(depth===0) return 0;
#   if(p.x<=5.0){
#     if(depth<3) return 0;
#     return p.y<=6.5?1:0;
#   } else {
#     if(depth<2) return 1;
#     return p.y<=5.0?1:0;
#   }
# }
#
# function regionColor(x,y,depth){
#   if(depth===0) return hex(C.blue,0.07);
#   if(x<=5.0){
#     if(depth<3) return hex(C.blue,0.07);
#     return y<=6.5?hex(C.orange,0.07):hex(C.blue,0.07);
#   } else {
#     if(depth<2) return hex(C.orange,0.07);
#     return y<=5.0?hex(C.orange,0.07):hex(C.blue,0.07);
#   }
# }
#
# function renderIntro(){
#   var depth=S.dtDepth;
#   var splits=SPLITS[depth-1];
#
#   /* accuracy */
#   var correct=INTRO_PTS.filter(function(p){return classifyPoint(p,depth)===p.c;}).length;
#   var acc=(correct/INTRO_PTS.length*100).toFixed(0);
#   var accCol=acc>90?C.green:acc>75?C.yellow:C.red;
#
#   /* build SVG */
#   var sv=plotAxes('Feature x\u2081','Feature x\u2082');
#
#   /* region shading — sample grid */
#   var step=1;
#   for(var gx=0;gx<10;gx+=step){
#     for(var gy=0;gy<10;gy+=step){
#       var cx2=gx+step/2, cy2=gy+step/2;
#       sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
#          +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
#          +' fill="'+regionColor(cx2,cy2,depth)+'"/>';
#     }
#   }
#
#   /* split lines */
#   var splitColors=[C.accent,C.yellow,C.purple];
#   splits.forEach(function(sp,i){
#     var col=splitColors[i];
#     if(sp.dir==='vert'){
#       var lx=sx(sp.val);
#       sv+='<line x1="'+lx.toFixed(1)+'" y1="'+PT+'" x2="'+lx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
#       sv+='<rect x="'+(lx-52)+'" y="'+(PT+3)+'" width="100" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
#       sv+='<text x="'+lx.toFixed(1)+'" y="'+(PT+13)+'" text-anchor="middle" fill="'+col+'" font-size="9" font-family="monospace" font-weight="700">x\u2081 \u2264 '+sp.val+'?</text>';
#     } else {
#       var ly=sy(sp.val);
#       var rx1=sx(sp.region[0]), rx2=sx(sp.region[1]);
#       sv+='<line x1="'+rx1.toFixed(1)+'" y1="'+ly.toFixed(1)+'" x2="'+rx2.toFixed(1)+'" y2="'+ly.toFixed(1)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
#       sv+='<rect x="'+(rx2-110)+'" y="'+(ly-16)+'" width="108" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
#       sv+='<text x="'+(rx2-6)+'" y="'+(ly-5)+'" text-anchor="end" fill="'+col+'" font-size="9" font-family="monospace" font-weight="700">x\u2082 \u2264 '+sp.val+'?</text>';
#     }
#   });
#
#   /* data points */
#   INTRO_PTS.forEach(function(p){
#     var pred=classifyPoint(p,depth);
#     var ok=pred===p.c;
#     var col=p.c===0?C.blue:C.orange;
#     sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5" fill="'+col+'"'
#        +' stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'"/>';
#     if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="10">\u2717</text>';
#   });
#
#   /* accuracy badge */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-22)+'" width="120" height="18" rx="4" fill="#0a0a0f" opacity="0.9"/>';
#   sv+='<text x="'+(PL+10)+'" y="'+(PT+PH-8)+'" fill="'+accCol+'" font-size="10" font-family="monospace" font-weight="700">accuracy: '+acc+'%</text>';
#
#   var out=sectionTitle('What is a Decision Tree?','A hierarchy of axis-aligned yes/no questions that partitions feature space into regions');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   /* left: interactive plot */
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Axis-Aligned Splits')
#     +svgBox(sv)
#     +'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
#     +[{c:C.blue,l:'Class 0'},{c:C.orange,l:'Class 1'},{c:C.red,l:'Misclassified'}].map(function(i){
#       return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
#         +'<div style="width:10px;height:10px;border-radius:50%;background:'+i.c+';"></div>'+i.l+'</div>';
#     }).join('')
#     +'</div>'
#     +sliderRow('dtDepth',depth,1,3,1,'tree depth',0)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.yellow+';">\u2190 coarse (1 split)</span>'
#     +'<span style="color:'+C.green+';">fine (3 splits) \u2192</span></div>'
#   );
#   out+='</div>';
#
#   /* right: concepts */
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT TREE');
#   out+=statRow('Depth',depth,C.accent);
#   out+=statRow('Splits applied',depth,C.yellow);
#   out+=statRow('Leaf regions',depth+1,C.purple);
#   out+=statRow('Accuracy',acc+'%',accCol);
#   out+=statRow('Class 0 correct',INTRO_PTS.filter(function(p){return p.c===0&&classifyPoint(p,depth)===0;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===0;}).length,C.blue);
#   out+=statRow('Class 1 correct',INTRO_PTS.filter(function(p){return p.c===1&&classifyPoint(p,depth)===1;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===1;}).length,C.orange);
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY CONCEPTS');
#   [
#     {lbl:'No linearity assumption',c:C.green},
#     {lbl:'Axis-aligned boundaries only',c:C.yellow},
#     {lbl:'Greedy recursive splitting',c:C.blue},
#     {lbl:'Each leaf = one prediction',c:C.purple},
#     {lbl:'Depth controls complexity',c:C.orange},
#   ].forEach(function(row){
#     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<div style="width:8px;height:8px;border-radius:2px;background:'+row.c+';flex-shrink:0;"></div>'
#       +'<span style="color:'+C.muted+';">'+row.lbl+'</span></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VS LINEAR MODELS');
#   [
#     {lbl:'Linear / LR / SVM',eq:'f(x) = w\u00b7x + b',desc:'features contribute additively',c:C.blue},
#     {lbl:'Decision Tree',eq:'f(x) = lookup leaf(x)',desc:'features contribute conditionally',c:C.accent},
#   ].forEach(function(m){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:6px;border-left:3px solid '+m.c+';">'
#       +'<div style="font-size:9px;font-weight:700;color:'+m.c+';">'+m.lbl+'</div>'
#       +'<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';margin:1px 0;">'+m.eq+'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';">'+m.desc+'</div></div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#9889;','Why Trees Can Model Anything',
#     'Each additional depth level doubles the number of leaf regions. '
#     +'A tree of depth d creates up to <span style="color:'+C.yellow+';font-family:monospace;">2\u1d48</span> rectangular regions — '
#     +'together they can approximate <span style="color:'+C.accent+';font-weight:700;">any</span> decision boundary, '
#     +'no matter how non-linear. The price: depth without constraint leads to <span style="color:'+C.red+';font-weight:700;">overfitting</span>.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 1 — IMPURITY CRITERIA (GINI VS ENTROPY)
# ══════════════════════════════════════════════════════ */
# function gini(p){return 2*p*(1-p);}
# function entropy(p){
#   if(p<=0||p>=1) return 0;
#   return -p*Math.log2(p)-(1-p)*Math.log2(1-p);
# }
#
# function renderImpurity(){
#   var p=S.pVal;
#   var gv=gini(p), ev=entropy(p);
#
#   /* build impurity curve chart */
#   var sv=plotAxes('p (fraction class 1)','Impurity',1,1,
#     [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#     [0,0.2,0.4,0.6,0.8,1.0]);
#
#   /* shade under curves */
#   var gPath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
#   var ePath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
#   var steps=100;
#   for(var i=0;i<=steps;i++){
#     var pp=i/steps;
#     gPath+=' L'+sx(pp,1).toFixed(1)+','+sy(gini(pp),1).toFixed(1);
#     ePath+=' L'+sx(pp,1).toFixed(1)+','+sy(entropy(pp),1).toFixed(1);
#   }
#   var baseY=sy(0,1);
#   gPath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
#   ePath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
#   sv+='<path d="'+gPath+'" fill="'+hex(C.orange,0.12)+'"/>';
#   sv+='<path d="'+ePath+'" fill="'+hex(C.blue,0.10)+'"/>';
#
#   /* curve lines */
#   var gLine='', eLine='';
#   for(var j=0;j<=steps;j++){
#     var pj=j/steps;
#     var gpt=sx(pj,1).toFixed(1)+','+sy(gini(pj),1).toFixed(1);
#     var ept=sx(pj,1).toFixed(1)+','+sy(entropy(pj),1).toFixed(1);
#     gLine+=(j===0?'M':'L')+gpt+' ';
#     eLine+=(j===0?'M':'L')+ept+' ';
#   }
#   sv+='<path d="'+gLine+'" fill="none" stroke="'+C.orange+'" stroke-width="2.2"/>';
#   sv+='<path d="'+eLine+'" fill="none" stroke="'+C.blue+'" stroke-width="2.2"/>';
#
#   /* vertical cursor */
#   var cx3=sx(p,1);
#   sv+='<line x1="'+cx3.toFixed(1)+'" y1="'+PT+'" x2="'+cx3.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
#   /* dots on curves */
#   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(gv,1).toFixed(1)+'" r="5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(ev,1).toFixed(1)+'" r="5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#
#   /* value labels */
#   var lx=cx3+8;
#   sv+='<rect x="'+(lx-2)+'" y="'+(sy(gv,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
#   sv+='<text x="'+lx+'" y="'+(sy(gv,1)+0)+'" fill="'+C.orange+'" font-size="9" font-family="monospace">Gini='+gv.toFixed(3)+'</text>';
#   sv+='<rect x="'+(lx-2)+'" y="'+(sy(ev,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
#   sv+='<text x="'+lx+'" y="'+(sy(ev,1)+0)+'" fill="'+C.blue+'" font-size="9" font-family="monospace">H='+ev.toFixed(3)+'</text>';
#
#   /* p label */
#   sv+='<text x="'+cx3.toFixed(1)+'" y="'+(PT+PH+26)+'" text-anchor="middle" fill="'+C.accent+'" font-size="9" font-weight="700">p='+p.toFixed(2)+'</text>';
#
#   /* legend */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="138" height="36" rx="4" fill="#0a0a0f" opacity="0.88"/>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+13)+'" x2="'+(PL+30)+'" y2="'+(PT+13)+'" stroke="'+C.orange+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+16)+'" fill="'+C.orange+'" font-size="9" font-family="monospace">Gini = 2p(1\u2212p)</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+28)+'" x2="'+(PL+30)+'" y2="'+(PT+28)+'" stroke="'+C.blue+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+31)+'" fill="'+C.blue+'" font-size="9" font-family="monospace">H = \u2212p\u00b7log\u2082p\u2026</text>';
#
#   /* ─── Gain demo: candidate split ─── */
#   var nParent=10, nPos=Math.round(p*nParent), nNeg=nParent-nPos;
#   var gParent=gini(p), hParent=entropy(p);
#   /* split: left 5/10, right 5/10 */
#   var nL=5, pL=Math.min(1,Math.max(0,Math.round(p*1.2*nL)/nL));
#   var nR=5, pR=Math.min(1,Math.max(0,Math.round(p*0.8*nR)/nR));
#   pL=Math.min(1,pL); pR=Math.min(1,pR);
#   var wGini=(nL/nParent)*gini(pL)+(nR/nParent)*gini(pR);
#   var wH   =(nL/nParent)*entropy(pL)+(nR/nParent)*entropy(pR);
#   var gGain=Math.max(0,gParent-wGini);
#   var igGain=Math.max(0,hParent-wH);
#   var gainCol=gGain>0.1?C.green:gGain>0.02?C.yellow:C.red;
#
#   var out=sectionTitle('Impurity Criteria','Gini and Entropy measure how mixed the class labels are at each node');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Impurity vs Class Proportion p')
#     +svgBox(sv)
#     +sliderRow('pVal',p,0.01,0.99,0.01,'p (class 1)',2)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.green+';">\u2190 pure class 0 (impurity = 0)</span>'
#     +'<span style="color:'+C.green+';">pure class 1 \u2192</span></div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#
#   /* current impurity values */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT p = '+p.toFixed(2));
#   out+=statRow('Gini impurity',''+gv.toFixed(4),C.orange);
#   out+=statRow('Entropy H',''+ev.toFixed(4)+' bits',C.blue);
#   out+=statRow('Is pure?',(p<=0.01||p>=0.99)?'Yes \u2713':'No',(p<=0.01||p>=0.99)?C.green:C.red);
#   out+=statRow('Max impurity',(Math.abs(p-0.5)<0.05)?'Yes \u2248 0.5':'No',(Math.abs(p-0.5)<0.05)?C.yellow:C.muted);
#   out+='</div>';
#
#   /* formulas */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','FORMULAS');
#   [
#     {lbl:'Gini (binary)',    eq:'1 \u2212 p\u00b2 \u2212 (1\u2212p)\u00b2',  sub:'= 2p(1\u2212p)',  c:C.orange},
#     {lbl:'Entropy (binary)', eq:'\u2212p\u00b7log\u2082p \u2212 (1\u2212p)\u00b7log\u2082(1\u2212p)', sub:'max at p=0.5: H=1 bit',c:C.blue},
#     {lbl:'Gini general',     eq:'1 \u2212 \u2211\u2c7c p\u2c7c\u00b2',          sub:'k classes',         c:C.yellow},
#   ].forEach(function(f){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+f.c+';">'
#       +'<div style="font-size:8.5px;color:'+C.dim+';">'+f.lbl+'</div>'
#       +'<div style="font-size:9.5px;font-family:monospace;color:'+f.c+';margin:2px 0;">'+f.eq+'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';">'+f.sub+'</div></div>';
#   });
#   out+='</div>';
#
#   /* information gain demo */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INFORMATION GAIN DEMO');
#   out+='<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Parent: '+nPos+' pos / '+nNeg+' neg (p='+p.toFixed(2)+')</div>';
#   [
#     {lbl:'Gini(parent)',   val:gParent.toFixed(4), c:C.orange},
#     {lbl:'Weighted Gini',  val:wGini.toFixed(4),   c:C.orange},
#     {lbl:'Gini Gain',      val:gGain.toFixed(4),   c:gainCol},
#     {lbl:'H(parent)',      val:hParent.toFixed(4)+' bits',c:C.blue},
#     {lbl:'Info Gain (IG)', val:igGain.toFixed(4)+' bits', c:gainCol},
#   ].forEach(function(r){out+=statRow(r.lbl,r.val,r.c);});
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#128200;','Gini vs Entropy in Practice',
#     'Gini is <span style="color:'+C.green+';font-weight:700;">slightly faster</span> (no log) and is the sklearn default. '
#     +'Entropy is <span style="color:'+C.yellow+';font-weight:700;">more sensitive near pure nodes</span> and can prefer different splits on ambiguous data. '
#     +'In practice, <span style="color:'+C.accent+';font-weight:700;">both produce nearly identical trees</span> — the choice rarely changes final accuracy by more than ~1%.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 2 — CART GROWING PROCESS
# ══════════════════════════════════════════════════════ */
# var CART_STEPS=[
#   {
#     title:'Step 0 — All data at root',
#     desc:'All 10 training examples land at the root. We compute the parent Gini impurity (6 Yes, 4 No \u2192 p=0.6).',
#     detail:'Gini(parent) = 1 \u2212 0.6\u00b2 \u2212 0.4\u00b2 = 1 \u2212 0.36 \u2212 0.16 = 0.48',
#     splitLabel:null,
#     nodes:[
#       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true}
#     ],
#     edges:[]
#   },
#   {
#     title:'Step 1 — Evaluate candidate splits',
#     desc:'For each feature and threshold, compute weighted child Gini. Find the split with maximum Gain.',
#     detail:'Age \u2264 42: Gain = 0.02 (small)\nIncome = High: Gain = 0.48 \u2714 (maximum!)',
#     splitLabel:'Best split: Income = High',
#     nodes:[
#       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true},
#       {id:'cA',x:100,y:128,w:130,h:40,label:'Income = High',sub:'Gini = 0.48 (Age \u226442)',col:C.red,active:false,dim:true},
#       {id:'cB',x:280,y:128,w:130,h:40,label:'Income = High',sub:'Gain = 0.48 \u2713 BEST',col:C.green,active:true},
#     ],
#     edges:[
#       {from:'root',to:'cA',label:'Age\u226442',col:C.dim},
#       {from:'root',to:'cB',label:'Income',col:C.green}
#     ]
#   },
#   {
#     title:'Step 2 — Apply best split',
#     desc:'Split on Income = High. Left child (High): 6 Yes, 0 No — PURE! Right child (Low): 0 Yes, 4 No — PURE!',
#     detail:'Weighted Gini = (6/10)\u00b70 + (4/10)\u00b70 = 0.00  \u2192  Gini Gain = 0.48',
#     splitLabel:'Both children are pure leaves \u2713',
#     nodes:[
#       {id:'root',x:220,y:22,w:170,h:40,label:'Income = High?',sub:'Gain=0.48 | n=10',col:C.accent,active:false},
#       {id:'left',x:100,y:120,w:140,h:46,label:'\u2714 High (Yes)',sub:'6 Yes, 0 No\nGini = 0.00 (PURE)',col:C.green,active:true},
#       {id:'right',x:300,y:120,w:140,h:46,label:'\u2718 Low (No)',sub:'0 Yes, 4 No\nGini = 0.00 (PURE)',col:C.green,active:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'Yes (High)',col:C.green},
#       {from:'root',to:'right',label:'No (Low)', col:C.orange}
#     ]
#   },
#   {
#     title:'Step 3 — Stopping conditions checked',
#     desc:'Both children are pure (Gini = 0). Recursion stops. Leaves are assigned majority class predictions.',
#     detail:'Stopping rule: node is pure \u2192 create leaf\nOther rules: max_depth reached, min_samples_split, zero gain',
#     splitLabel:'Tree fully grown \u2014 depth = 1',
#     nodes:[
#       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'root node | n=10',col:C.accent,active:false},
#       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES',sub:'Leaf | Gini=0 | n=6',col:C.green,active:true,leaf:true},
#       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | Gini=0 | n=4',col:C.blue,active:true,leaf:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'High',col:C.green},
#       {from:'root',to:'right',label:'Low', col:C.orange}
#     ]
#   },
#   {
#     title:'Step 4 — Making a prediction',
#     desc:'New example: Age=35, Income=High. Route from root \u2192 follow "High" branch \u2192 reach leaf \u2192 Predict: YES.',
#     detail:'Runtime: O(depth) = O(1) for this tree.\nOnly the path root\u2192leaf is evaluated \u2014 all other nodes are ignored.',
#     splitLabel:'Prediction in O(depth) time',
#     nodes:[
#       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'New example: Income=High',col:C.yellow,active:true},
#       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES \u2714',sub:'New example lands here',col:C.green,active:true,leaf:true,highlight:true},
#       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | n=4',col:C.blue,active:false,leaf:true},
#     ],
#     edges:[
#       {from:'root',to:'left', label:'\u2714 High \u2192 here',col:C.yellow},
#       {from:'root',to:'right',label:'Low',col:C.dim}
#     ]
#   }
# ];
#
# function treeNode(n){
#   var fill=n.dim?hex(C.card,0.5):(n.highlight?hex(n.col,0.25):hex(n.col,0.12));
#   var border=n.dim?C.dim:(n.active?n.col:C.border);
#   var bw=n.active?2:1;
#   var lines=n.sub.split('\n');
#   return '<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'"'
#     +' rx="7" fill="'+fill+'" stroke="'+border+'" stroke-width="'+bw+'"/>'
#     +(n.leaf?'<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="4" rx="2" fill="'+border+'"/>':'')
#     +'<text x="'+n.x+'" y="'+(n.y+14)+'" text-anchor="middle" fill="'+(n.dim?C.dim:n.col)+'" font-size="10" font-family="monospace" font-weight="700">'+n.label+'</text>'
#     +lines.map(function(l,i){
#       return '<text x="'+n.x+'" y="'+(n.y+27+i*11)+'" text-anchor="middle" fill="'+(n.dim?C.dim:C.muted)+'" font-size="8" font-family="monospace">'+l+'</text>';
#     }).join('');
# }
#
# function treeEdge(e,nodes){
#   var fromNode=nodes.find(function(n){return n.id===e.from;});
#   var toNode=nodes.find(function(n){return n.id===e.to;});
#   if(!fromNode||!toNode) return '';
#   var x1=fromNode.x, y1=fromNode.y+fromNode.h;
#   var x2=toNode.x, y2=toNode.y;
#   var mx=(x1+x2)/2, my=(y1+y2)/2;
#   return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+e.col+'" stroke-width="1.5"/>'
#     +'<rect x="'+(mx-28)+'" y="'+(my-8)+'" width="56" height="12" rx="3" fill="#0a0a0f" opacity="0.85"/>'
#     +'<text x="'+mx+'" y="'+(my+2)+'" text-anchor="middle" fill="'+e.col+'" font-size="8.5" font-family="monospace">'+e.label+'</text>';
# }
#
# function renderCART(){
#   var step=S.cartStep;
#   var cs=CART_STEPS[step];
#
#   var svInner='';
#   cs.edges.forEach(function(e){ svInner+=treeEdge(e,cs.nodes); });
#   cs.nodes.forEach(function(n){ svInner+=treeNode(n); });
#
#   var out=sectionTitle('CART Growing Process','The greedy recursive algorithm that builds decision trees one split at a time');
#
#   /* step navigation */
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
#   CART_STEPS.forEach(function(st,i){
#     var on=i===step;
#     out+='<button data-action="cartStep" data-idx="'+i+'"'
#       +' style="padding:7px 12px;border-radius:8px;font-size:9.5px;font-weight:700;font-family:inherit;'
#       +'background:'+(on?hex(C.accent,.15):C.card)+';border:1.5px solid '+(on?C.accent:C.border)+';'
#       +'color:'+(on?C.accent:C.muted)+';cursor:pointer;transition:all .2s;">'
#       +'Step '+i+'</button>';
#   });
#   out+='</div>';
#
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   /* tree diagram */
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',cs.title)
#     +svgBox(svInner,440,200)
#   );
#   out+='</div>';
#
#   /* step explanation */
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:10px;','WHAT\'S HAPPENING');
#   out+=div('font-size:9.5px;color:'+C.text+';line-height:1.8;margin-bottom:10px;',cs.desc);
#   out+='<div style="background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';">'
#     +'<pre style="font-size:8.5px;color:'+C.muted+';white-space:pre-wrap;line-height:1.7;font-family:monospace;">'+cs.detail+'</pre>'
#     +'</div>';
#   if(cs.splitLabel){
#     out+='<div style="margin-top:10px;padding:8px 12px;background:'+hex(C.green,.08)+';border-radius:6px;border:1px solid '+hex(C.green,.3)+';font-size:9.5px;font-weight:700;color:'+C.green+';">'
#       +'\u2714\ufe0f '+cs.splitLabel+'</div>';
#   }
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','STOPPING CONDITIONS');
#   [
#     {c:C.green,  lbl:'Node is pure (Gini=0)', active:step>=2},
#     {c:C.yellow, lbl:'max_depth reached',     active:false},
#     {c:C.blue,   lbl:'min_samples_split',     active:false},
#     {c:C.orange, lbl:'Zero gain split',       active:false},
#   ].forEach(function(r){
#     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<div style="width:8px;height:8px;border-radius:50%;background:'+(r.active?r.c:C.dim)+';flex-shrink:0;"></div>'
#       +'<span style="color:'+(r.active?r.c:C.dim)+';">'+r.lbl+(r.active?' \u2714':'')+'</span></div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   /* navigation buttons */
#   out+='<div style="display:flex;gap:10px;justify-content:center;margin-bottom:12px;">';
#   out+='<button data-action="cartPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(step>0?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
#     +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Prev</button>';
#   out+='<button data-action="cartNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
#     +'background:'+(step<4?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
#     +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next \u2192</button>';
#   out+='</div>';
#
#   out+=insight('&#128296;','Why Greedy? Why Not Global?',
#     'Finding the <span style="color:'+C.red+';font-weight:700;">globally optimal tree</span> is NP-hard — '
#     +'the number of possible trees grows super-exponentially with depth and features. '
#     +'CART\'s <span style="color:'+C.accent+';font-weight:700;">greedy approach</span> makes locally optimal decisions at each node in O(np log n) time. '
#     +'The greedy solution is good enough in practice — '
#     +'and ensembles like <span style="color:'+C.yellow+';font-weight:700;">Random Forests</span> compensate for its suboptimality.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 3 — OVERFITTING & PRUNING
# ══════════════════════════════════════════════════════ */
# /* Simulated train/test error curves as function of depth */
# function trainError(d){
#   return Math.max(0, 0.42 * Math.exp(-0.55*d));
# }
# function testError(d){
#   /* U-shaped: sweet spot near depth 3-4 */
#   return 0.08 + 0.25*Math.exp(-0.6*(d-1)) + 0.018*(d-3)*(d-3);
# }
# /* After pruning, test error improves for deep trees */
# function prunedTestError(d,alpha){
#   var base=testError(d);
#   var improvement=Math.min(base*0.6, alpha*(d-2)*0.04);
#   return Math.max(0.07, base-improvement);
# }
#
# function renderOverfit(){
#   var depth=S.treeDepth;
#   var alpha=S.pruneAlpha;
#   var trE=trainError(depth), teE=testError(depth), prE=prunedTestError(depth,alpha);
#   var sweetSpot=3;
#
#   /* error curve plot */
#   var sv=plotAxes('Tree Depth','Error',8,0.5,
#     [1,2,3,4,5,6,7,8],[0,0.1,0.2,0.3,0.4,0.5]);
#
#   /* sweet spot band */
#   sv+='<rect x="'+sx(2.6,8).toFixed(1)+'" y="'+PT+'" width="'+(sx(3.8,8)-sx(2.6,8)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.green,0.06)+'"/>';
#   sv+='<text x="'+sx(3.2,8).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.green+'" font-size="8.5" font-family="monospace">sweet spot</text>';
#
#   /* train error curve */
#   var trnPath='', tstPath='', prnPath='';
#   for(var d=1;d<=8;d+=0.1){
#     var te=trainError(d), te2=testError(d), pe=prunedTestError(d,alpha);
#     var px=sx(d,8), pyt=sy(te,0.5), pyte=sy(te2,0.5), pype=sy(pe,0.5);
#     trnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyt.toFixed(1)+' ';
#     tstPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyte.toFixed(1)+' ';
#     prnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pype.toFixed(1)+' ';
#   }
#   sv+='<path d="'+trnPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
#   sv+='<path d="'+tstPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';
#   if(alpha>0.01) sv+='<path d="'+prnPath+'" fill="none" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
#
#   /* current depth vertical */
#   var dvx=sx(depth,8);
#   sv+='<line x1="'+dvx.toFixed(1)+'" y1="'+PT+'" x2="'+dvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
#   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(trE,0.5).toFixed(1)+'" r="4" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(teE,0.5).toFixed(1)+'" r="4" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#   if(alpha>0.01) sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(prE,0.5).toFixed(1)+'" r="4" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="1.5"/>';
#
#   /* legend */
#   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-56)+'" width="150" height="54" rx="4" fill="#0a0a0f" opacity="0.9"/>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-44)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-44)+'" stroke="'+C.green+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-41)+'" fill="'+C.green+'" font-size="9" font-family="monospace">Train error</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-30)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-30)+'" stroke="'+C.red+'" stroke-width="2"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-27)+'" fill="'+C.red+'" font-size="9" font-family="monospace">Test error</text>';
#   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-16)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-16)+'" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
#   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-13)+'" fill="'+C.yellow+'" font-size="9" font-family="monospace">Pruned test</text>';
#
#   var bias=depth<=2?'High':'Low';
#   var variance=depth>=5?'High':'Low';
#   var regime=depth<=2?'Underfitting':depth>=6?'Overfitting':'Good fit';
#   var regimeCol=depth<=2?C.blue:depth>=6?C.red:C.green;
#
#   var out=sectionTitle('Overfitting & Pruning','Depth controls the bias-variance tradeoff — constrain or prune to prevent memorisation');
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Train vs Test Error by Depth')
#     +svgBox(sv)
#     +sliderRow('treeDepth',depth,1,8,1,'max_depth',0)
#     +sliderRow('pruneAlpha',alpha,0,0.5,0.01,'ccp_alpha',2)
#     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
#     +'<span style="color:'+C.blue+';">\u2190 underfitting</span>'
#     +'<span style="color:'+C.red+';">overfitting \u2192</span></div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT STATE');
#   out+=statRow('max_depth',depth,C.accent);
#   out+=statRow('Max leaf nodes',Math.pow(2,depth),C.purple);
#   out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
#   out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
#   if(alpha>0.01) out+=statRow('Pruned test',(prE*100).toFixed(1)+'%',C.yellow);
#   out+=statRow('Bias',bias,depth<=2?C.orange:C.green);
#   out+=statRow('Variance',variance,depth>=5?C.orange:C.green);
#   out+=statRow('Regime',regime,regimeCol);
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRE-PRUNING PARAMS');
#   [
#     {p:'max_depth',     desc:'max root-to-leaf distance',    eg:'3\u20135',    c:C.accent},
#     {p:'min_samples_split', desc:'min samples to split node', eg:'\u226510',   c:C.yellow},
#     {p:'min_samples_leaf',  desc:'min samples in each leaf',  eg:'\u22655',    c:C.blue},
#     {p:'min_impurity_decrease', desc:'only split if gain \u2265 this',eg:'0.01',c:C.purple},
#   ].forEach(function(r){
#     out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
#       +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
#       +'<span style="color:'+r.c+';font-family:monospace;font-weight:700;">'+r.p+'</span>'
#       +'<span style="color:'+C.muted+';">e.g. '+r.eg+'</span></div>'
#       +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','POST-PRUNING (ccp_alpha)');
#   out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
#     'Grows full tree, then removes subtrees with penalty:<br>'
#     +'<span style="color:'+C.accent+';font-family:monospace;">R\u03b1(T) = R(T) + \u03b1 \u00b7 |T|</span><br>'
#     +'Higher \u03b1 = simpler tree. Find optimal \u03b1 via cross-validation.<br>'
#     +'sklearn: <span style="color:'+C.yellow+';font-family:monospace;">ccp_alpha</span> parameter.'
#   );
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#127919;','The U-Curve Rule of Thumb',
#     'Training error <span style="color:'+C.green+';font-weight:700;">monotonically decreases</span> with depth — a depth-n tree can perfectly memorise n training points. '
#     +'Test error follows a <span style="color:'+C.red+';font-weight:700;">U-curve</span>. '
#     +'Start with <span style="color:'+C.accent+';font-family:monospace;">max_depth=3</span> and increase if the model underfits. '
#     +'Use <span style="color:'+C.yellow+';font-family:monospace;">cross-validation</span> to find the sweet spot — never judge by training error alone.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    TAB 4 — FEATURE IMPORTANCE (MDI)
# ══════════════════════════════════════════════════════ */
# var FEAT_DATASETS=[
#   {
#     name:'Medical Diagnosis',
#     features:['Blood Pressure','Cholesterol','Age','BMI','Glucose','Smoking'],
#     imps:[0.38,0.25,0.18,0.10,0.06,0.03],
#     cols:[C.red,C.orange,C.yellow,C.blue,C.purple,C.muted],
#     rootFeat:'Blood Pressure',
#     note:'BP is split at root (affects 100% of samples), giving ~2.5\u00d7 the importance of Cholesterol.'
#   },
#   {
#     name:'Customer Churn',
#     features:['Contract Type','Monthly Charges','Tenure','Tech Support','Payment Method','Gender'],
#     imps:[0.42,0.27,0.15,0.09,0.05,0.02],
#     cols:[C.accent,C.blue,C.green,C.yellow,C.orange,C.muted],
#     rootFeat:'Contract Type',
#     note:'Long-term contracts strongly predict retention. Gender has near-zero importance \u2014 as expected.'
#   }
# ];
#
# function renderFeatureImportance(){
#   var ds=FEAT_DATASETS[S.featDataset];
#   var imps=ds.imps;
#   var total=imps.reduce(function(a,b){return a+b;},0);
#   var norm=imps.map(function(v){return v/total;});
#
#   /* bar chart SVG */
#   var BW=420, BH=200;
#   var bPL=110, bPR=20, bPT=15, bPB=20;
#   var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
#   var barH=Math.floor(bPH/norm.length)-6;
#
#   var sv='';
#   /* gridlines */
#   [0,0.1,0.2,0.3,0.4].forEach(function(v){
#     var bx=bPL+v*bPW;
#     sv+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(bPT+bPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
#     sv+='<text x="'+bx.toFixed(1)+'" y="'+(bPT+bPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
#   });
#   sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
#   sv+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-3)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">MDI Importance (%)</text>';
#
#   norm.forEach(function(v,i){
#     var by=bPT+i*(barH+6);
#     var bw2=v*bPW;
#     var col=ds.cols[i];
#     /* bar bg */
#     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(col,0.08)+'"/>';
#     /* bar fill */
#     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(col,0.75)+'"/>';
#     /* rank badge if top 1 */
#     if(i===0){
#       sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3"'
#          +' fill="none" stroke="'+col+'" stroke-width="1.5"/>';
#       sv+='<rect x="'+(bPL+bw2+4)+'" y="'+(by+2)+'" width="30" height="'+(barH-4)+'" rx="3" fill="'+hex(col,0.2)+'" stroke="'+col+'" stroke-width="1"/>';
#       sv+='<text x="'+(bPL+bw2+19)+'" y="'+(by+barH/2+3)+'" text-anchor="middle" fill="'+col+'" font-size="8" font-weight="700">ROOT</text>';
#     }
#     /* feature label */
#     sv+='<text x="'+(bPL-6)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="'+(i<3?C.text:C.muted)+'" font-size="9" font-family="monospace">'+ds.features[i]+'</text>';
#     /* value */
#     sv+='<text x="'+(bPL+bw2-4)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="#0a0a0f" font-size="8.5" font-weight="700">'+(v*100).toFixed(1)+'%</text>';
#   });
#
#   /* MDI formula annotation */
#   var formulaX=bPL+bPW*0.55, formulaY=bPT+10;
#   sv+='<rect x="'+formulaX.toFixed(1)+'" y="'+formulaY.toFixed(1)+'" width="140" height="30" rx="4" fill="#0a0a0f" opacity="0.92" stroke="'+C.border+'" stroke-width="1"/>';
#   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+11).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">Importance(f) =</text>';
#   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+23).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="8" font-family="monospace">\u03a3 (|node|/n)\u00b7Gain(node)</text>';
#
#   var out=sectionTitle('Feature Importance (MDI)','Mean Decrease in Impurity — how much each feature reduces node impurity across all splits');
#
#   out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
#   FEAT_DATASETS.forEach(function(d,i){
#     out+=btnSel(i,S.featDataset,C.accent,'&#128202; '+d.name,'featData');
#   });
#   out+='</div>';
#
#   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
#
#   out+='<div style="flex:1 1 380px;">';
#   out+=card(
#     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',ds.name+' \u2014 Feature Importances')
#     +svgBox(sv,BW,BH)
#     +'<div style="margin-top:10px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
#     +'<div style="font-size:8.5px;color:'+C.muted+';line-height:1.7;">Root feature: <span style="color:'+ds.cols[0]+';font-weight:700;">'+ds.rootFeat+'</span> \u2014 '+ds.note+'</div>'
#     +'</div>'
#   );
#   out+='</div>';
#
#   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
#
#   /* top features table */
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TOP FEATURES');
#   norm.forEach(function(v,i){
#     var bar=Math.round(v*20);
#     out+='<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
#       +'<div style="width:16px;text-align:right;font-size:9px;color:'+C.dim+';">'+(i+1)+'.</div>'
#       +'<div style="flex:1;font-size:9px;color:'+(i<2?C.text:C.muted)+';">'+ds.features[i]+'</div>'
#       +'<div style="font-size:9px;font-family:monospace;color:'+ds.cols[i]+';font-weight:700;width:36px;text-align:right;">'+(v*100).toFixed(1)+'%</div>'
#       +'</div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHAT MDI TELLS YOU');
#   [
#     {icon:'\u2714',lbl:'Features tree uses most',c:C.green},
#     {icon:'\u2714',lbl:'Relative training influence',c:C.green},
#     {icon:'\u2718',lbl:'Direction of relationship',c:C.red},
#     {icon:'\u2718',lbl:'Causal importance',c:C.red},
#     {icon:'\u26a0',lbl:'Biased toward high-cardinality features',c:C.yellow},
#   ].forEach(function(r){
#     out+='<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;font-size:9px;">'
#       +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
#       +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
#   });
#   out+='</div>';
#
#   out+='<div class="card" style="margin:0;">';
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MDI vs PERMUTATION');
#   [
#     {lbl:'MDI (default)',        pro:'fast, no refit needed',   con:'biased to high cardinality', c:C.orange},
#     {lbl:'Permutation',          pro:'unbiased, model-agnostic', con:'slower (n_repeats \u00d7 fit)', c:C.blue},
#   ].forEach(function(r){
#     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
#       +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
#       +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
#       +'<div style="font-size:8px;color:'+C.red+';margin-top:1px;">\u2718 '+r.con+'</div>'
#       +'</div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   out+=insight('&#9888;&#65039;','MDI Bias Caveat',
#     'MDI is <span style="color:'+C.red+';font-weight:700;">biased toward features with many unique values</span> — '
#     +'they have more threshold candidates, increasing their chance of a high gain by chance. '
#     +'For unbiased importance use <span style="color:'+C.accent+';font-family:monospace;">sklearn.inspection.permutation_importance</span> '
#     +'or consider <span style="color:'+C.yellow+';font-weight:700;">SHAP values</span> for both direction and magnitude of influence.'
#   );
#   return out;
# }
#
# /* ══════════════════════════════════════════════════════
#    ROOT RENDER
# ══════════════════════════════════════════════════════ */
# var TABS=[
#   '\u26d4 Intro &amp; Splits',
#   '&#128202; Impurity Criteria',
#   '&#127794; CART Growing',
#   '\u2702\uFE0F Overfitting &amp; Pruning',
#   '&#128161; Feature Importance'
# ];
#
# function renderApp(){
#   var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
#   html+='<div style="text-align:center;margin-bottom:16px;">'
#     +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.yellow+','+C.orange+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Decision Tree</div>'
#     +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough \u2014 from axis-aligned splits and impurity to CART, pruning and feature importance')
#     +'</div>';
#   html+='<div class="tab-bar">';
#   TABS.forEach(function(t,i){
#     html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
#   });
#   html+='</div>';
#   html+='<div class="fade">';
#   if(S.tab===0)      html+=renderIntro();
#   else if(S.tab===1) html+=renderImpurity();
#   else if(S.tab===2) html+=renderCART();
#   else if(S.tab===3) html+=renderOverfit();
#   else if(S.tab===4) html+=renderFeatureImportance();
#   html+='</div></div>';
#   return html;
# }
#
# function render(){
#   document.getElementById('app').innerHTML=renderApp();
#   bindEvents();
# }
#
# function bindEvents(){
#   document.querySelectorAll('[data-action]').forEach(function(el){
#     var action=el.getAttribute('data-action');
#     var idx=parseInt(el.getAttribute('data-idx'));
#     var tag=el.tagName.toLowerCase();
#     if(tag==='button'){
#       el.addEventListener('click',function(){
#         if(action==='tab')        {S.tab=idx; render();}
#         else if(action==='cartStep'){S.cartStep=idx; render();}
#         else if(action==='cartNext'){if(S.cartStep<4){S.cartStep++;render();}}
#         else if(action==='cartPrev'){if(S.cartStep>0){S.cartStep--;render();}}
#         else if(action==='featData'){S.featDataset=idx; render();}
#       });
#     } else if(tag==='input'){
#       el.addEventListener('input',function(){
#         var val=parseFloat(this.value);
#         if(action==='dtDepth')      {S.dtDepth=Math.round(val);  render();}
#         else if(action==='pVal')    {S.pVal=val;                 render();}
#         else if(action==='treeDepth'){S.treeDepth=Math.round(val);render();}
#         else if(action==='pruneAlpha'){S.pruneAlpha=val;         render();}
#       });
#     }
#   });
# }
#
# render();
# </script>
# </body>
# </html>"""
#
# DT_VISUAL_HEIGHT = 1100
#
# # """
# # Self-contained HTML visual for Decision Trees.
# # 5 interactive tabs: Intro & Axis-Aligned Splits, Impurity Criteria (Gini vs Entropy),
# # CART Growing Process, Overfitting & Pruning, Feature Importance.
# # Pure vanilla HTML/JS — zero CDN dependencies.
# # Embed via: st.components.v1.html(DT_VISUAL_HTML, height=DT_VISUAL_HEIGHT, scrolling=True)
# # """
# #
# # DT_VISUAL_HTML = r"""<!DOCTYPE html>
# # <html>
# # <head>
# # <meta charset="utf-8"/>
# # <style>
# # *{margin:0;padding:0;box-sizing:border-box;}
# # body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
# # button{cursor:pointer;font-family:inherit;}
# # input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
# # input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
# # @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
# # .fade{animation:fadeIn .3s ease both;}
# # .card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
# # .tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
# # .tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
# # .tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
# # .section-title{text-align:center;margin-bottom:20px;}
# # .section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
# # .section-title p{font-size:11px;color:#71717a;}
# # .insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
# # .ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
# # .ins-body{font-size:11px;color:#71717a;line-height:1.8;}
# # </style>
# # </head>
# # <body>
# # <div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
# # <script>
# # /* ─── PALETTE ─── */
# # var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
# #   accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
# #   yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
# #   dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
# #   orange:"#fb923c"};
# #
# # /* ─── HELPERS ─── */
# # function hex(c,a){
# #   var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
# #   return 'rgba('+r+','+g+','+b+','+a+')';
# # }
# # function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
# # function card(inner,extra){
# #   return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
# # }
# # function sectionTitle(t,s){
# #   return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
# # }
# # function insight(icon,title,body){
# #   return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
# # }
# # function btnSel(idx,cur,color,label,action){
# #   var on=idx===cur;
# #   return '<button data-action="'+action+'" data-idx="'+idx
# #     +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
# #     +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
# #     +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
# # }
# # function sliderRow(action,val,min,max,step,label,dec){
# #   var dv=(dec!==undefined)?val.toFixed(dec):val;
# #   return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
# #     +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
# #     +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
# #     +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
# #     +'</div>';
# # }
# # function statRow(label,val,color){
# #   return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
# #     +'<span style="color:'+C.muted+';">'+label+'</span>'
# #     +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
# # }
# # function svgBox(inner,w,h){
# #   return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
# # }
# #
# # /* ─── SVG PLOT SCAFFOLD ─── */
# # var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
# # var PW=VW-PL-PR, PH=VH-PT-PB;
# # function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
# # function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
# # function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
# #   var xm=xmax||10, ym=ymax||10;
# #   var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
# #   var o='';
# #   xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
# #   yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
# #   o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
# #   o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
# #   xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
# #   yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
# #   o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
# #   o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
# #   return o;
# # }
# #
# # /* ─── STATE ─── */
# # var S={
# #   tab:0,
# #   dtDepth:1,      /* intro split depth 1..3 */
# #   pVal:0.5,       /* impurity p slider */
# #   cartStep:0,     /* 0..4 growing steps */
# #   treeDepth:1,    /* overfitting depth 1..8 */
# #   pruneAlpha:0,   /* ccp_alpha 0..0.5 */
# #   featDataset:0   /* 0=synthetic A  1=synthetic B */
# # };
# #
# # /* ══════════════════════════════════════════════════════
# #    TAB 0 — INTRO & AXIS-ALIGNED SPLITS
# # ══════════════════════════════════════════════════════ */
# # /* Fixed 2-class dataset with two clusters */
# # var INTRO_PTS=[
# #   /* class 0 (blue) — upper-left cluster */
# #   {x:1.5,y:7.5,c:0},{x:2.2,y:8.8,c:0},{x:3.0,y:7.0,c:0},{x:1.8,y:9.2,c:0},
# #   {x:2.8,y:6.2,c:0},{x:1.2,y:6.8,c:0},{x:3.5,y:8.5,c:0},{x:4.2,y:7.8,c:0},
# #   /* class 1 (orange) — lower-right cluster */
# #   {x:6.5,y:2.5,c:1},{x:7.8,y:3.8,c:1},{x:8.5,y:1.8,c:1},{x:7.2,y:1.2,c:1},
# #   {x:9.0,y:3.2,c:1},{x:6.0,y:4.5,c:1},{x:8.0,y:4.8,c:1},{x:9.5,y:2.0,c:1},
# #   /* overlap region */
# #   {x:4.8,y:5.5,c:0},{x:5.5,y:4.8,c:1},{x:5.2,y:6.2,c:0},{x:6.2,y:5.8,c:1}
# # ];
# #
# # /* Precomputed splits for depth 1,2,3 */
# # var SPLITS=[
# #   /* depth 1 */
# #   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]}],
# #   /* depth 2 — add horizontal cut in right half */
# #   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
# #    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]}],
# #   /* depth 3 — refine left side too */
# #   [{feat:'x\u2081',thresh:5.0,dir:'vert', val:5.0, region:[0,10,0,10]},
# #    {feat:'x\u2082',thresh:5.0,dir:'horiz',val:5.0, region:[5,10,0,10]},
# #    {feat:'x\u2082',thresh:6.5,dir:'horiz',val:6.5, region:[0,5,0,10]}]
# # ];
# #
# # function classifyPoint(p,depth){
# #   /* route point through learned axis-aligned splits */
# #   if(depth===0) return 0;
# #   if(p.x<=5.0){
# #     if(depth<3) return 0;
# #     return p.y<=6.5?1:0;
# #   } else {
# #     if(depth<2) return 1;
# #     return p.y<=5.0?1:0;
# #   }
# # }
# #
# # function regionColor(x,y,depth){
# #   if(depth===0) return hex(C.blue,0.07);
# #   if(x<=5.0){
# #     if(depth<3) return hex(C.blue,0.07);
# #     return y<=6.5?hex(C.orange,0.07):hex(C.blue,0.07);
# #   } else {
# #     if(depth<2) return hex(C.orange,0.07);
# #     return y<=5.0?hex(C.orange,0.07):hex(C.blue,0.07);
# #   }
# # }
# #
# # function renderIntro(){
# #   var depth=S.dtDepth;
# #   var splits=SPLITS.slice(0,depth);
# #
# #   /* accuracy */
# #   var correct=INTRO_PTS.filter(function(p){return classifyPoint(p,depth)===p.c;}).length;
# #   var acc=(correct/INTRO_PTS.length*100).toFixed(0);
# #   var accCol=acc>90?C.green:acc>75?C.yellow:C.red;
# #
# #   /* build SVG */
# #   var sv=plotAxes('Feature x\u2081','Feature x\u2082');
# #
# #   /* region shading — sample grid */
# #   var step=1;
# #   for(var gx=0;gx<10;gx+=step){
# #     for(var gy=0;gy<10;gy+=step){
# #       var cx2=gx+step/2, cy2=gy+step/2;
# #       sv+='<rect x="'+sx(gx).toFixed(1)+'" y="'+sy(gy+step).toFixed(1)+'"'
# #          +' width="'+(sx(gx+step)-sx(gx)).toFixed(1)+'" height="'+(sy(gy)-sy(gy+step)).toFixed(1)+'"'
# #          +' fill="'+regionColor(cx2,cy2,depth)+'"/>';
# #     }
# #   }
# #
# #   /* split lines */
# #   var splitColors=[C.accent,C.yellow,C.purple];
# #   splits.forEach(function(sp,i){
# #     var col=splitColors[i];
# #     if(sp.dir==='vert'){
# #       var lx=sx(sp.val);
# #       sv+='<line x1="'+lx.toFixed(1)+'" y1="'+PT+'" x2="'+lx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
# #       sv+='<rect x="'+(lx-52)+'" y="'+(PT+3)+'" width="100" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
# #       sv+='<text x="'+lx.toFixed(1)+'" y="'+(PT+13)+'" text-anchor="middle" fill="'+col+'" font-size="9" font-family="monospace" font-weight="700">x\u2081 \u2264 '+sp.val+'?</text>';
# #     } else {
# #       var ly=sy(sp.val);
# #       var rx1=sx(sp.region[0]), rx2=sx(sp.region[1]);
# #       sv+='<line x1="'+rx1.toFixed(1)+'" y1="'+ly.toFixed(1)+'" x2="'+rx2.toFixed(1)+'" y2="'+ly.toFixed(1)+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="6,3"/>';
# #       sv+='<rect x="'+(rx2-110)+'" y="'+(ly-16)+'" width="108" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
# #       sv+='<text x="'+(rx2-6)+'" y="'+(ly-5)+'" text-anchor="end" fill="'+col+'" font-size="9" font-family="monospace" font-weight="700">x\u2082 \u2264 '+sp.val+'?</text>';
# #     }
# #   });
# #
# #   /* data points */
# #   INTRO_PTS.forEach(function(p){
# #     var pred=classifyPoint(p,depth);
# #     var ok=pred===p.c;
# #     var col=p.c===0?C.blue:C.orange;
# #     sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5" fill="'+col+'"'
# #        +' stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'"/>';
# #     if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="10">\u2717</text>';
# #   });
# #
# #   /* accuracy badge */
# #   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-22)+'" width="120" height="18" rx="4" fill="#0a0a0f" opacity="0.9"/>';
# #   sv+='<text x="'+(PL+10)+'" y="'+(PT+PH-8)+'" fill="'+accCol+'" font-size="10" font-family="monospace" font-weight="700">accuracy: '+acc+'%</text>';
# #
# #   var out=sectionTitle('What is a Decision Tree?','A hierarchy of axis-aligned yes/no questions that partitions feature space into regions');
# #   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
# #
# #   /* left: interactive plot */
# #   out+='<div style="flex:1 1 380px;">';
# #   out+=card(
# #     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Axis-Aligned Splits')
# #     +svgBox(sv)
# #     +'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">'
# #     +[{c:C.blue,l:'Class 0'},{c:C.orange,l:'Class 1'},{c:C.red,l:'Misclassified'}].map(function(i){
# #       return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
# #         +'<div style="width:10px;height:10px;border-radius:50%;background:'+i.c+';"></div>'+i.l+'</div>';
# #     }).join('')
# #     +'</div>'
# #     +sliderRow('dtDepth',depth,1,3,1,'tree depth',0)
# #     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
# #     +'<span style="color:'+C.yellow+';">\u2190 coarse (1 split)</span>'
# #     +'<span style="color:'+C.green+';">fine (3 splits) \u2192</span></div>'
# #   );
# #   out+='</div>';
# #
# #   /* right: concepts */
# #   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT TREE');
# #   out+=statRow('Depth',depth,C.accent);
# #   out+=statRow('Splits applied',depth,C.yellow);
# #   out+=statRow('Leaf regions',depth+1,C.purple);
# #   out+=statRow('Accuracy',acc+'%',accCol);
# #   out+=statRow('Class 0 correct',INTRO_PTS.filter(function(p){return p.c===0&&classifyPoint(p,depth)===0;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===0;}).length,C.blue);
# #   out+=statRow('Class 1 correct',INTRO_PTS.filter(function(p){return p.c===1&&classifyPoint(p,depth)===1;}).length+' / '+INTRO_PTS.filter(function(p){return p.c===1;}).length,C.orange);
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY CONCEPTS');
# #   [
# #     {lbl:'No linearity assumption',c:C.green},
# #     {lbl:'Axis-aligned boundaries only',c:C.yellow},
# #     {lbl:'Greedy recursive splitting',c:C.blue},
# #     {lbl:'Each leaf = one prediction',c:C.purple},
# #     {lbl:'Depth controls complexity',c:C.orange},
# #   ].forEach(function(row){
# #     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
# #       +'<div style="width:8px;height:8px;border-radius:2px;background:'+row.c+';flex-shrink:0;"></div>'
# #       +'<span style="color:'+C.muted+';">'+row.lbl+'</span></div>';
# #   });
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VS LINEAR MODELS');
# #   [
# #     {lbl:'Linear / LR / SVM',eq:'f(x) = w\u00b7x + b',desc:'features contribute additively',c:C.blue},
# #     {lbl:'Decision Tree',eq:'f(x) = lookup leaf(x)',desc:'features contribute conditionally',c:C.accent},
# #   ].forEach(function(m){
# #     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:6px;border-left:3px solid '+m.c+';">'
# #       +'<div style="font-size:9px;font-weight:700;color:'+m.c+';">'+m.lbl+'</div>'
# #       +'<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';margin:1px 0;">'+m.eq+'</div>'
# #       +'<div style="font-size:8px;color:'+C.dim+';">'+m.desc+'</div></div>';
# #   });
# #   out+='</div>';
# #   out+='</div></div>';
# #
# #   out+=insight('&#9889;','Why Trees Can Model Anything',
# #     'Each additional depth level doubles the number of leaf regions. '
# #     +'A tree of depth d creates up to <span style="color:'+C.yellow+';font-family:monospace;">2\u1d48</span> rectangular regions — '
# #     +'together they can approximate <span style="color:'+C.accent+';font-weight:700;">any</span> decision boundary, '
# #     +'no matter how non-linear. The price: depth without constraint leads to <span style="color:'+C.red+';font-weight:700;">overfitting</span>.'
# #   );
# #   return out;
# # }
# #
# # /* ══════════════════════════════════════════════════════
# #    TAB 1 — IMPURITY CRITERIA (GINI VS ENTROPY)
# # ══════════════════════════════════════════════════════ */
# # function gini(p){return 2*p*(1-p);}
# # function entropy(p){
# #   if(p<=0||p>=1) return 0;
# #   return -p*Math.log2(p)-(1-p)*Math.log2(1-p);
# # }
# #
# # function renderImpurity(){
# #   var p=S.pVal;
# #   var gv=gini(p), ev=entropy(p);
# #
# #   /* build impurity curve chart */
# #   var sv=plotAxes('p (fraction class 1)','Impurity',1,1,
# #     [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
# #     [0,0.2,0.4,0.6,0.8,1.0]);
# #
# #   /* shade under curves */
# #   var gPath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
# #   var ePath='M'+sx(0,1).toFixed(1)+','+sy(0,1).toFixed(1);
# #   var steps=100;
# #   for(var i=0;i<=steps;i++){
# #     var pp=i/steps;
# #     gPath+=' L'+sx(pp,1).toFixed(1)+','+sy(gini(pp),1).toFixed(1);
# #     ePath+=' L'+sx(pp,1).toFixed(1)+','+sy(entropy(pp),1).toFixed(1);
# #   }
# #   var baseY=sy(0,1);
# #   gPath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
# #   ePath+=' L'+sx(1,1).toFixed(1)+','+baseY+' Z';
# #   sv+='<path d="'+gPath+'" fill="'+hex(C.orange,0.12)+'"/>';
# #   sv+='<path d="'+ePath+'" fill="'+hex(C.blue,0.10)+'"/>';
# #
# #   /* curve lines */
# #   var gLine='', eLine='';
# #   for(var j=0;j<=steps;j++){
# #     var pj=j/steps;
# #     var gpt=sx(pj,1).toFixed(1)+','+sy(gini(pj),1).toFixed(1);
# #     var ept=sx(pj,1).toFixed(1)+','+sy(entropy(pj),1).toFixed(1);
# #     gLine+=(j===0?'M':'L')+gpt+' ';
# #     eLine+=(j===0?'M':'L')+ept+' ';
# #   }
# #   sv+='<path d="'+gLine+'" fill="none" stroke="'+C.orange+'" stroke-width="2.2"/>';
# #   sv+='<path d="'+eLine+'" fill="none" stroke="'+C.blue+'" stroke-width="2.2"/>';
# #
# #   /* vertical cursor */
# #   var cx3=sx(p,1);
# #   sv+='<line x1="'+cx3.toFixed(1)+'" y1="'+PT+'" x2="'+cx3.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
# #   /* dots on curves */
# #   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(gv,1).toFixed(1)+'" r="5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';
# #   sv+='<circle cx="'+cx3.toFixed(1)+'" cy="'+sy(ev,1).toFixed(1)+'" r="5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
# #
# #   /* value labels */
# #   var lx=cx3+8;
# #   sv+='<rect x="'+(lx-2)+'" y="'+(sy(gv,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
# #   sv+='<text x="'+lx+'" y="'+(sy(gv,1)+0)+'" fill="'+C.orange+'" font-size="9" font-family="monospace">Gini='+gv.toFixed(3)+'</text>';
# #   sv+='<rect x="'+(lx-2)+'" y="'+(sy(ev,1)-10)+'" width="66" height="12" rx="2" fill="#0a0a0f" opacity="0.85"/>';
# #   sv+='<text x="'+lx+'" y="'+(sy(ev,1)+0)+'" fill="'+C.blue+'" font-size="9" font-family="monospace">H='+ev.toFixed(3)+'</text>';
# #
# #   /* p label */
# #   sv+='<text x="'+cx3.toFixed(1)+'" y="'+(PT+PH+26)+'" text-anchor="middle" fill="'+C.accent+'" font-size="9" font-weight="700">p='+p.toFixed(2)+'</text>';
# #
# #   /* legend */
# #   sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="138" height="36" rx="4" fill="#0a0a0f" opacity="0.88"/>';
# #   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+13)+'" x2="'+(PL+30)+'" y2="'+(PT+13)+'" stroke="'+C.orange+'" stroke-width="2"/>';
# #   sv+='<text x="'+(PL+34)+'" y="'+(PT+16)+'" fill="'+C.orange+'" font-size="9" font-family="monospace">Gini = 2p(1\u2212p)</text>';
# #   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+28)+'" x2="'+(PL+30)+'" y2="'+(PT+28)+'" stroke="'+C.blue+'" stroke-width="2"/>';
# #   sv+='<text x="'+(PL+34)+'" y="'+(PT+31)+'" fill="'+C.blue+'" font-size="9" font-family="monospace">H = \u2212p\u00b7log\u2082p\u2026</text>';
# #
# #   /* ─── Gain demo: candidate split ─── */
# #   var nParent=10, nPos=Math.round(p*nParent), nNeg=nParent-nPos;
# #   var gParent=gini(p), hParent=entropy(p);
# #   /* split: left 5/10, right 5/10 */
# #   var nL=5, pL=Math.min(1,Math.max(0,Math.round(p*1.2*nL)/nL));
# #   var nR=5, pR=Math.min(1,Math.max(0,Math.round(p*0.8*nR)/nR));
# #   pL=Math.min(1,pL); pR=Math.min(1,pR);
# #   var wGini=(nL/nParent)*gini(pL)+(nR/nParent)*gini(pR);
# #   var wH   =(nL/nParent)*entropy(pL)+(nR/nParent)*entropy(pR);
# #   var gGain=Math.max(0,gParent-wGini);
# #   var igGain=Math.max(0,hParent-wH);
# #   var gainCol=gGain>0.1?C.green:gGain>0.02?C.yellow:C.red;
# #
# #   var out=sectionTitle('Impurity Criteria','Gini and Entropy measure how mixed the class labels are at each node');
# #   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
# #
# #   out+='<div style="flex:1 1 380px;">';
# #   out+=card(
# #     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Impurity vs Class Proportion p')
# #     +svgBox(sv)
# #     +sliderRow('pVal',p,0.01,0.99,0.01,'p (class 1)',2)
# #     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
# #     +'<span style="color:'+C.green+';">\u2190 pure class 0 (impurity = 0)</span>'
# #     +'<span style="color:'+C.green+';">pure class 1 \u2192</span></div>'
# #   );
# #   out+='</div>';
# #
# #   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
# #
# #   /* current impurity values */
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT p = '+p.toFixed(2));
# #   out+=statRow('Gini impurity',''+gv.toFixed(4),C.orange);
# #   out+=statRow('Entropy H',''+ev.toFixed(4)+' bits',C.blue);
# #   out+=statRow('Is pure?',(p<=0.01||p>=0.99)?'Yes \u2713':'No',(p<=0.01||p>=0.99)?C.green:C.red);
# #   out+=statRow('Max impurity',(Math.abs(p-0.5)<0.05)?'Yes \u2248 0.5':'No',(Math.abs(p-0.5)<0.05)?C.yellow:C.muted);
# #   out+='</div>';
# #
# #   /* formulas */
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','FORMULAS');
# #   [
# #     {lbl:'Gini (binary)',    eq:'1 \u2212 p\u00b2 \u2212 (1\u2212p)\u00b2',  sub:'= 2p(1\u2212p)',  c:C.orange},
# #     {lbl:'Entropy (binary)', eq:'\u2212p\u00b7log\u2082p \u2212 (1\u2212p)\u00b7log\u2082(1\u2212p)', sub:'max at p=0.5: H=1 bit',c:C.blue},
# #     {lbl:'Gini general',     eq:'1 \u2212 \u2211\u2c7c p\u2c7c\u00b2',          sub:'k classes',         c:C.yellow},
# #   ].forEach(function(f){
# #     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+f.c+';">'
# #       +'<div style="font-size:8.5px;color:'+C.dim+';">'+f.lbl+'</div>'
# #       +'<div style="font-size:9.5px;font-family:monospace;color:'+f.c+';margin:2px 0;">'+f.eq+'</div>'
# #       +'<div style="font-size:8px;color:'+C.dim+';">'+f.sub+'</div></div>';
# #   });
# #   out+='</div>';
# #
# #   /* information gain demo */
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INFORMATION GAIN DEMO');
# #   out+='<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Parent: '+nPos+' pos / '+nNeg+' neg (p='+p.toFixed(2)+')</div>';
# #   [
# #     {lbl:'Gini(parent)',   val:gParent.toFixed(4), c:C.orange},
# #     {lbl:'Weighted Gini',  val:wGini.toFixed(4),   c:C.orange},
# #     {lbl:'Gini Gain',      val:gGain.toFixed(4),   c:gainCol},
# #     {lbl:'H(parent)',      val:hParent.toFixed(4)+' bits',c:C.blue},
# #     {lbl:'Info Gain (IG)', val:igGain.toFixed(4)+' bits', c:gainCol},
# #   ].forEach(function(r){out+=statRow(r.lbl,r.val,r.c);});
# #   out+='</div>';
# #   out+='</div></div>';
# #
# #   out+=insight('&#128200;','Gini vs Entropy in Practice',
# #     'Gini is <span style="color:'+C.green+';font-weight:700;">slightly faster</span> (no log) and is the sklearn default. '
# #     +'Entropy is <span style="color:'+C.yellow+';font-weight:700;">more sensitive near pure nodes</span> and can prefer different splits on ambiguous data. '
# #     +'In practice, <span style="color:'+C.accent+';font-weight:700;">both produce nearly identical trees</span> — the choice rarely changes final accuracy by more than ~1%.'
# #   );
# #   return out;
# # }
# #
# # /* ══════════════════════════════════════════════════════
# #    TAB 2 — CART GROWING PROCESS
# # ══════════════════════════════════════════════════════ */
# # var CART_STEPS=[
# #   {
# #     title:'Step 0 — All data at root',
# #     desc:'All 10 training examples land at the root. We compute the parent Gini impurity (6 Yes, 4 No \u2192 p=0.6).',
# #     detail:'Gini(parent) = 1 \u2212 0.6\u00b2 \u2212 0.4\u00b2 = 1 \u2212 0.36 \u2212 0.16 = 0.48',
# #     splitLabel:null,
# #     nodes:[
# #       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true}
# #     ],
# #     edges:[]
# #   },
# #   {
# #     title:'Step 1 — Evaluate candidate splits',
# #     desc:'For each feature and threshold, compute weighted child Gini. Find the split with maximum Gain.',
# #     detail:'Age \u2264 42: Gain = 0.02 (small)\nIncome = High: Gain = 0.48 \u2714 (maximum!)',
# #     splitLabel:'Best split: Income = High',
# #     nodes:[
# #       {id:'root',x:220,y:30,w:170,h:46,label:'Root Node',sub:'6 Yes, 4 No | Gini=0.48',col:C.yellow,active:true},
# #       {id:'cA',x:100,y:128,w:130,h:40,label:'Income = High',sub:'Gini = 0.48 (Age \u226442)',col:C.red,active:false,dim:true},
# #       {id:'cB',x:280,y:128,w:130,h:40,label:'Income = High',sub:'Gain = 0.48 \u2713 BEST',col:C.green,active:true},
# #     ],
# #     edges:[
# #       {from:'root',to:'cA',label:'Age\u226442',col:C.dim},
# #       {from:'root',to:'cB',label:'Income',col:C.green}
# #     ]
# #   },
# #   {
# #     title:'Step 2 — Apply best split',
# #     desc:'Split on Income = High. Left child (High): 6 Yes, 0 No — PURE! Right child (Low): 0 Yes, 4 No — PURE!',
# #     detail:'Weighted Gini = (6/10)\u00b70 + (4/10)\u00b70 = 0.00  \u2192  Gini Gain = 0.48',
# #     splitLabel:'Both children are pure leaves \u2713',
# #     nodes:[
# #       {id:'root',x:220,y:22,w:170,h:40,label:'Income = High?',sub:'Gain=0.48 | n=10',col:C.accent,active:false},
# #       {id:'left',x:100,y:120,w:140,h:46,label:'\u2714 High (Yes)',sub:'6 Yes, 0 No\nGini = 0.00 (PURE)',col:C.green,active:true},
# #       {id:'right',x:300,y:120,w:140,h:46,label:'\u2718 Low (No)',sub:'0 Yes, 4 No\nGini = 0.00 (PURE)',col:C.green,active:true},
# #     ],
# #     edges:[
# #       {from:'root',to:'left', label:'Yes (High)',col:C.green},
# #       {from:'root',to:'right',label:'No (Low)', col:C.orange}
# #     ]
# #   },
# #   {
# #     title:'Step 3 — Stopping conditions checked',
# #     desc:'Both children are pure (Gini = 0). Recursion stops. Leaves are assigned majority class predictions.',
# #     detail:'Stopping rule: node is pure \u2192 create leaf\nOther rules: max_depth reached, min_samples_split, zero gain',
# #     splitLabel:'Tree fully grown \u2014 depth = 1',
# #     nodes:[
# #       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'root node | n=10',col:C.accent,active:false},
# #       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES',sub:'Leaf | Gini=0 | n=6',col:C.green,active:true,leaf:true},
# #       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | Gini=0 | n=4',col:C.blue,active:true,leaf:true},
# #     ],
# #     edges:[
# #       {from:'root',to:'left', label:'High',col:C.green},
# #       {from:'root',to:'right',label:'Low', col:C.orange}
# #     ]
# #   },
# #   {
# #     title:'Step 4 — Making a prediction',
# #     desc:'New example: Age=35, Income=High. Route from root \u2192 follow "High" branch \u2192 reach leaf \u2192 Predict: YES.',
# #     detail:'Runtime: O(depth) = O(1) for this tree.\nOnly the path root\u2192leaf is evaluated \u2014 all other nodes are ignored.',
# #     splitLabel:'Prediction in O(depth) time',
# #     nodes:[
# #       {id:'root',x:220,y:18,w:168,h:40,label:'Income = High?',sub:'New example: Income=High',col:C.yellow,active:true},
# #       {id:'left',x:90,y:112,w:140,h:46,label:'Predict: YES \u2714',sub:'New example lands here',col:C.green,active:true,leaf:true,highlight:true},
# #       {id:'right',x:300,y:112,w:140,h:46,label:'Predict: NO',sub:'Leaf | n=4',col:C.blue,active:false,leaf:true},
# #     ],
# #     edges:[
# #       {from:'root',to:'left', label:'\u2714 High \u2192 here',col:C.yellow},
# #       {from:'root',to:'right',label:'Low',col:C.dim}
# #     ]
# #   }
# # ];
# #
# # function treeNode(n){
# #   var fill=n.dim?hex(C.card,0.5):(n.highlight?hex(n.col,0.25):hex(n.col,0.12));
# #   var border=n.dim?C.dim:(n.active?n.col:C.border);
# #   var bw=n.active?2:1;
# #   var lines=n.sub.split('\n');
# #   return '<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="'+n.h+'"'
# #     +' rx="7" fill="'+fill+'" stroke="'+border+'" stroke-width="'+bw+'"/>'
# #     +(n.leaf?'<rect x="'+(n.x-n.w/2)+'" y="'+n.y+'" width="'+n.w+'" height="4" rx="2" fill="'+border+'"/>':'')
# #     +'<text x="'+n.x+'" y="'+(n.y+14)+'" text-anchor="middle" fill="'+(n.dim?C.dim:n.col)+'" font-size="10" font-family="monospace" font-weight="700">'+n.label+'</text>'
# #     +lines.map(function(l,i){
# #       return '<text x="'+n.x+'" y="'+(n.y+27+i*11)+'" text-anchor="middle" fill="'+(n.dim?C.dim:C.muted)+'" font-size="8" font-family="monospace">'+l+'</text>';
# #     }).join('');
# # }
# #
# # function treeEdge(e,nodes){
# #   var fromNode=nodes.find(function(n){return n.id===e.from;});
# #   var toNode=nodes.find(function(n){return n.id===e.to;});
# #   if(!fromNode||!toNode) return '';
# #   var x1=fromNode.x, y1=fromNode.y+fromNode.h;
# #   var x2=toNode.x, y2=toNode.y;
# #   var mx=(x1+x2)/2, my=(y1+y2)/2;
# #   return '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+e.col+'" stroke-width="1.5"/>'
# #     +'<rect x="'+(mx-28)+'" y="'+(my-8)+'" width="56" height="12" rx="3" fill="#0a0a0f" opacity="0.85"/>'
# #     +'<text x="'+mx+'" y="'+(my+2)+'" text-anchor="middle" fill="'+e.col+'" font-size="8.5" font-family="monospace">'+e.label+'</text>';
# # }
# #
# # function renderCART(){
# #   var step=S.cartStep;
# #   var cs=CART_STEPS[step];
# #
# #   var svInner='';
# #   cs.edges.forEach(function(e){ svInner+=treeEdge(e,cs.nodes); });
# #   cs.nodes.forEach(function(n){ svInner+=treeNode(n); });
# #
# #   var out=sectionTitle('CART Growing Process','The greedy recursive algorithm that builds decision trees one split at a time');
# #
# #   /* step navigation */
# #   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
# #   CART_STEPS.forEach(function(st,i){
# #     var on=i===step;
# #     out+='<button data-action="cartStep" data-idx="'+i+'"'
# #       +' style="padding:7px 12px;border-radius:8px;font-size:9.5px;font-weight:700;font-family:inherit;'
# #       +'background:'+(on?hex(C.accent,.15):C.card)+';border:1.5px solid '+(on?C.accent:C.border)+';'
# #       +'color:'+(on?C.accent:C.muted)+';cursor:pointer;transition:all .2s;">'
# #       +'Step '+i+'</button>';
# #   });
# #   out+='</div>';
# #
# #   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
# #
# #   /* tree diagram */
# #   out+='<div style="flex:1 1 380px;">';
# #   out+=card(
# #     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',cs.title)
# #     +svgBox(svInner,440,200)
# #   );
# #   out+='</div>';
# #
# #   /* step explanation */
# #   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:10px;','WHAT\'S HAPPENING');
# #   out+=div('font-size:9.5px;color:'+C.text+';line-height:1.8;margin-bottom:10px;',cs.desc);
# #   out+='<div style="background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';">'
# #     +'<pre style="font-size:8.5px;color:'+C.muted+';white-space:pre-wrap;line-height:1.7;font-family:monospace;">'+cs.detail+'</pre>'
# #     +'</div>';
# #   if(cs.splitLabel){
# #     out+='<div style="margin-top:10px;padding:8px 12px;background:'+hex(C.green,.08)+';border-radius:6px;border:1px solid '+hex(C.green,.3)+';font-size:9.5px;font-weight:700;color:'+C.green+';">'
# #       +'\u2714\ufe0f '+cs.splitLabel+'</div>';
# #   }
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','STOPPING CONDITIONS');
# #   [
# #     {c:C.green,  lbl:'Node is pure (Gini=0)', active:step>=2},
# #     {c:C.yellow, lbl:'max_depth reached',     active:false},
# #     {c:C.blue,   lbl:'min_samples_split',     active:false},
# #     {c:C.orange, lbl:'Zero gain split',       active:false},
# #   ].forEach(function(r){
# #     out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
# #       +'<div style="width:8px;height:8px;border-radius:50%;background:'+(r.active?r.c:C.dim)+';flex-shrink:0;"></div>'
# #       +'<span style="color:'+(r.active?r.c:C.dim)+';">'+r.lbl+(r.active?' \u2714':'')+'</span></div>';
# #   });
# #   out+='</div>';
# #   out+='</div></div>';
# #
# #   /* navigation buttons */
# #   out+='<div style="display:flex;gap:10px;justify-content:center;margin-bottom:12px;">';
# #   out+='<button data-action="cartPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
# #     +'background:'+(step>0?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
# #     +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Prev</button>';
# #   out+='<button data-action="cartNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
# #     +'background:'+(step<4?hex(C.accent,.12):C.card)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
# #     +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Next \u2192</button>';
# #   out+='</div>';
# #
# #   out+=insight('&#128296;','Why Greedy? Why Not Global?',
# #     'Finding the <span style="color:'+C.red+';font-weight:700;">globally optimal tree</span> is NP-hard — '
# #     +'the number of possible trees grows super-exponentially with depth and features. '
# #     +'CART\'s <span style="color:'+C.accent+';font-weight:700;">greedy approach</span> makes locally optimal decisions at each node in O(np log n) time. '
# #     +'The greedy solution is good enough in practice — '
# #     +'and ensembles like <span style="color:'+C.yellow+';font-weight:700;">Random Forests</span> compensate for its suboptimality.'
# #   );
# #   return out;
# # }
# #
# # /* ══════════════════════════════════════════════════════
# #    TAB 3 — OVERFITTING & PRUNING
# # ══════════════════════════════════════════════════════ */
# # /* Simulated train/test error curves as function of depth */
# # function trainError(d){
# #   return Math.max(0, 0.42 * Math.exp(-0.55*d));
# # }
# # function testError(d){
# #   /* U-shaped: sweet spot near depth 3-4 */
# #   return 0.08 + 0.25*Math.exp(-0.6*(d-1)) + 0.018*(d-3)*(d-3);
# # }
# # /* After pruning, test error improves for deep trees */
# # function prunedTestError(d,alpha){
# #   var base=testError(d);
# #   var improvement=Math.min(base*0.6, alpha*(d-2)*0.04);
# #   return Math.max(0.07, base-improvement);
# # }
# #
# # function renderOverfit(){
# #   var depth=S.treeDepth;
# #   var alpha=S.pruneAlpha;
# #   var trE=trainError(depth), teE=testError(depth), prE=prunedTestError(depth,alpha);
# #   var sweetSpot=3;
# #
# #   /* error curve plot */
# #   var sv=plotAxes('Tree Depth','Error',8,0.5,
# #     [1,2,3,4,5,6,7,8],[0,0.1,0.2,0.3,0.4,0.5]);
# #
# #   /* sweet spot band */
# #   sv+='<rect x="'+sx(2.6,8).toFixed(1)+'" y="'+PT+'" width="'+(sx(3.8,8)-sx(2.6,8)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.green,0.06)+'"/>';
# #   sv+='<text x="'+sx(3.2,8).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.green+'" font-size="8.5" font-family="monospace">sweet spot</text>';
# #
# #   /* train error curve */
# #   var trnPath='', tstPath='', prnPath='';
# #   for(var d=1;d<=8;d+=0.1){
# #     var te=trainError(d), te2=testError(d), pe=prunedTestError(d,alpha);
# #     var px=sx(d,8), pyt=sy(te,0.5), pyte=sy(te2,0.5), pype=sy(pe,0.5);
# #     trnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyt.toFixed(1)+' ';
# #     tstPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pyte.toFixed(1)+' ';
# #     prnPath+=(d<=1.05?'M':'L')+px.toFixed(1)+','+pype.toFixed(1)+' ';
# #   }
# #   sv+='<path d="'+trnPath+'" fill="none" stroke="'+C.green+'" stroke-width="2"/>';
# #   sv+='<path d="'+tstPath+'" fill="none" stroke="'+C.red+'" stroke-width="2"/>';
# #   if(alpha>0.01) sv+='<path d="'+prnPath+'" fill="none" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
# #
# #   /* current depth vertical */
# #   var dvx=sx(depth,8);
# #   sv+='<line x1="'+dvx.toFixed(1)+'" y1="'+PT+'" x2="'+dvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
# #   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(trE,0.5).toFixed(1)+'" r="4" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
# #   sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(teE,0.5).toFixed(1)+'" r="4" fill="'+C.red+'" stroke="#0a0a0f" stroke-width="1.5"/>';
# #   if(alpha>0.01) sv+='<circle cx="'+dvx.toFixed(1)+'" cy="'+sy(prE,0.5).toFixed(1)+'" r="4" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="1.5"/>';
# #
# #   /* legend */
# #   sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-56)+'" width="150" height="54" rx="4" fill="#0a0a0f" opacity="0.9"/>';
# #   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-44)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-44)+'" stroke="'+C.green+'" stroke-width="2"/>';
# #   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-41)+'" fill="'+C.green+'" font-size="9" font-family="monospace">Train error</text>';
# #   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-30)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-30)+'" stroke="'+C.red+'" stroke-width="2"/>';
# #   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-27)+'" fill="'+C.red+'" font-size="9" font-family="monospace">Test error</text>';
# #   sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-16)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-16)+'" stroke="'+C.yellow+'" stroke-width="2" stroke-dasharray="5,3"/>';
# #   sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-13)+'" fill="'+C.yellow+'" font-size="9" font-family="monospace">Pruned test</text>';
# #
# #   var bias=depth<=2?'High':'Low';
# #   var variance=depth>=5?'High':'Low';
# #   var regime=depth<=2?'Underfitting':depth>=6?'Overfitting':'Good fit';
# #   var regimeCol=depth<=2?C.blue:depth>=6?C.red:C.green;
# #
# #   var out=sectionTitle('Overfitting & Pruning','Depth controls the bias-variance tradeoff — constrain or prune to prevent memorisation');
# #   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
# #
# #   out+='<div style="flex:1 1 380px;">';
# #   out+=card(
# #     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Train vs Test Error by Depth')
# #     +svgBox(sv)
# #     +sliderRow('treeDepth',depth,1,8,1,'max_depth',0)
# #     +sliderRow('pruneAlpha',alpha,0,0.5,0.01,'ccp_alpha',2)
# #     +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
# #     +'<span style="color:'+C.blue+';">\u2190 underfitting</span>'
# #     +'<span style="color:'+C.red+';">overfitting \u2192</span></div>'
# #   );
# #   out+='</div>';
# #
# #   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT STATE');
# #   out+=statRow('max_depth',depth,C.accent);
# #   out+=statRow('Max leaf nodes',Math.pow(2,depth),C.purple);
# #   out+=statRow('Train error',(trE*100).toFixed(1)+'%',C.green);
# #   out+=statRow('Test error',(teE*100).toFixed(1)+'%',C.red);
# #   if(alpha>0.01) out+=statRow('Pruned test',(prE*100).toFixed(1)+'%',C.yellow);
# #   out+=statRow('Bias',bias,depth<=2?C.orange:C.green);
# #   out+=statRow('Variance',variance,depth>=5?C.orange:C.green);
# #   out+=statRow('Regime',regime,regimeCol);
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRE-PRUNING PARAMS');
# #   [
# #     {p:'max_depth',     desc:'max root-to-leaf distance',    eg:'3\u20135',    c:C.accent},
# #     {p:'min_samples_split', desc:'min samples to split node', eg:'\u226510',   c:C.yellow},
# #     {p:'min_samples_leaf',  desc:'min samples in each leaf',  eg:'\u22655',    c:C.blue},
# #     {p:'min_impurity_decrease', desc:'only split if gain \u2265 this',eg:'0.01',c:C.purple},
# #   ].forEach(function(r){
# #     out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
# #       +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
# #       +'<span style="color:'+r.c+';font-family:monospace;font-weight:700;">'+r.p+'</span>'
# #       +'<span style="color:'+C.muted+';">e.g. '+r.eg+'</span></div>'
# #       +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
# #   });
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','POST-PRUNING (ccp_alpha)');
# #   out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
# #     'Grows full tree, then removes subtrees with penalty:<br>'
# #     +'<span style="color:'+C.accent+';font-family:monospace;">R\u03b1(T) = R(T) + \u03b1 \u00b7 |T|</span><br>'
# #     +'Higher \u03b1 = simpler tree. Find optimal \u03b1 via cross-validation.<br>'
# #     +'sklearn: <span style="color:'+C.yellow+';font-family:monospace;">ccp_alpha</span> parameter.'
# #   );
# #   out+='</div>';
# #   out+='</div></div>';
# #
# #   out+=insight('&#127919;','The U-Curve Rule of Thumb',
# #     'Training error <span style="color:'+C.green+';font-weight:700;">monotonically decreases</span> with depth — a depth-n tree can perfectly memorise n training points. '
# #     +'Test error follows a <span style="color:'+C.red+';font-weight:700;">U-curve</span>. '
# #     +'Start with <span style="color:'+C.accent+';font-family:monospace;">max_depth=3</span> and increase if the model underfits. '
# #     +'Use <span style="color:'+C.yellow+';font-family:monospace;">cross-validation</span> to find the sweet spot — never judge by training error alone.'
# #   );
# #   return out;
# # }
# #
# # /* ══════════════════════════════════════════════════════
# #    TAB 4 — FEATURE IMPORTANCE (MDI)
# # ══════════════════════════════════════════════════════ */
# # var FEAT_DATASETS=[
# #   {
# #     name:'Medical Diagnosis',
# #     features:['Blood Pressure','Cholesterol','Age','BMI','Glucose','Smoking'],
# #     imps:[0.38,0.25,0.18,0.10,0.06,0.03],
# #     cols:[C.red,C.orange,C.yellow,C.blue,C.purple,C.muted],
# #     rootFeat:'Blood Pressure',
# #     note:'BP is split at root (affects 100% of samples), giving ~2.5\u00d7 the importance of Cholesterol.'
# #   },
# #   {
# #     name:'Customer Churn',
# #     features:['Contract Type','Monthly Charges','Tenure','Tech Support','Payment Method','Gender'],
# #     imps:[0.42,0.27,0.15,0.09,0.05,0.02],
# #     cols:[C.accent,C.blue,C.green,C.yellow,C.orange,C.muted],
# #     rootFeat:'Contract Type',
# #     note:'Long-term contracts strongly predict retention. Gender has near-zero importance \u2014 as expected.'
# #   }
# # ];
# #
# # function renderFeatureImportance(){
# #   var ds=FEAT_DATASETS[S.featDataset];
# #   var imps=ds.imps;
# #   var total=imps.reduce(function(a,b){return a+b;},0);
# #   var norm=imps.map(function(v){return v/total;});
# #
# #   /* bar chart SVG */
# #   var BW=420, BH=200;
# #   var bPL=110, bPR=20, bPT=15, bPB=20;
# #   var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
# #   var barH=Math.floor(bPH/norm.length)-6;
# #
# #   var sv='';
# #   /* gridlines */
# #   [0,0.1,0.2,0.3,0.4].forEach(function(v){
# #     var bx=bPL+v*bPW;
# #     sv+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(bPT+bPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
# #     sv+='<text x="'+bx.toFixed(1)+'" y="'+(bPT+bPH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
# #   });
# #   sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
# #   sv+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-3)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">MDI Importance (%)</text>';
# #
# #   norm.forEach(function(v,i){
# #     var by=bPT+i*(barH+6);
# #     var bw2=v*bPW;
# #     var col=ds.cols[i];
# #     /* bar bg */
# #     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(col,0.08)+'"/>';
# #     /* bar fill */
# #     sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(col,0.75)+'"/>';
# #     /* rank badge if top 1 */
# #     if(i===0){
# #       sv+='<rect x="'+bPL+'" y="'+by+'" width="'+bw2.toFixed(1)+'" height="'+barH+'" rx="3"'
# #          +' fill="none" stroke="'+col+'" stroke-width="1.5"/>';
# #       sv+='<rect x="'+(bPL+bw2+4)+'" y="'+(by+2)+'" width="30" height="'+(barH-4)+'" rx="3" fill="'+hex(col,0.2)+'" stroke="'+col+'" stroke-width="1"/>';
# #       sv+='<text x="'+(bPL+bw2+19)+'" y="'+(by+barH/2+3)+'" text-anchor="middle" fill="'+col+'" font-size="8" font-weight="700">ROOT</text>';
# #     }
# #     /* feature label */
# #     sv+='<text x="'+(bPL-6)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="'+(i<3?C.text:C.muted)+'" font-size="9" font-family="monospace">'+ds.features[i]+'</text>';
# #     /* value */
# #     sv+='<text x="'+(bPL+bw2-4)+'" y="'+(by+barH/2+3)+'" text-anchor="end" fill="#0a0a0f" font-size="8.5" font-weight="700">'+(v*100).toFixed(1)+'%</text>';
# #   });
# #
# #   /* MDI formula annotation */
# #   var formulaX=bPL+bPW*0.55, formulaY=bPT+10;
# #   sv+='<rect x="'+formulaX.toFixed(1)+'" y="'+formulaY.toFixed(1)+'" width="140" height="30" rx="4" fill="#0a0a0f" opacity="0.92" stroke="'+C.border+'" stroke-width="1"/>';
# #   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+11).toFixed(1)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">Importance(f) =</text>';
# #   sv+='<text x="'+(formulaX+70).toFixed(1)+'" y="'+(formulaY+23).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="8" font-family="monospace">\u03a3 (|node|/n)\u00b7Gain(node)</text>';
# #
# #   var out=sectionTitle('Feature Importance (MDI)','Mean Decrease in Impurity — how much each feature reduces node impurity across all splits');
# #
# #   out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
# #   FEAT_DATASETS.forEach(function(d,i){
# #     out+=btnSel(i,S.featDataset,C.accent,'\u{1F4CA} '+d.name,'featData');
# #   });
# #   out+='</div>';
# #
# #   out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
# #
# #   out+='<div style="flex:1 1 380px;">';
# #   out+=card(
# #     div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',ds.name+' \u2014 Feature Importances')
# #     +svgBox(sv,BW,BH)
# #     +'<div style="margin-top:10px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
# #     +'<div style="font-size:8.5px;color:'+C.muted+';line-height:1.7;">Root feature: <span style="color:'+ds.cols[0]+';font-weight:700;">'+ds.rootFeat+'</span> \u2014 '+ds.note+'</div>'
# #     +'</div>'
# #   );
# #   out+='</div>';
# #
# #   out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
# #
# #   /* top features table */
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TOP FEATURES');
# #   norm.forEach(function(v,i){
# #     var bar=Math.round(v*20);
# #     out+='<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
# #       +'<div style="width:16px;text-align:right;font-size:9px;color:'+C.dim+';">'+(i+1)+'.</div>'
# #       +'<div style="flex:1;font-size:9px;color:'+(i<2?C.text:C.muted)+';">'+ds.features[i]+'</div>'
# #       +'<div style="font-size:9px;font-family:monospace;color:'+ds.cols[i]+';font-weight:700;width:36px;text-align:right;">'+(v*100).toFixed(1)+'%</div>'
# #       +'</div>';
# #   });
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHAT MDI TELLS YOU');
# #   [
# #     {icon:'\u2714',lbl:'Features tree uses most',c:C.green},
# #     {icon:'\u2714',lbl:'Relative training influence',c:C.green},
# #     {icon:'\u2718',lbl:'Direction of relationship',c:C.red},
# #     {icon:'\u2718',lbl:'Causal importance',c:C.red},
# #     {icon:'\u26a0',lbl:'Biased toward high-cardinality features',c:C.yellow},
# #   ].forEach(function(r){
# #     out+='<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;font-size:9px;">'
# #       +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
# #       +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
# #   });
# #   out+='</div>';
# #
# #   out+='<div class="card" style="margin:0;">';
# #   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MDI vs PERMUTATION');
# #   [
# #     {lbl:'MDI (default)',        pro:'fast, no refit needed',   con:'biased to high cardinality', c:C.orange},
# #     {lbl:'Permutation',          pro:'unbiased, model-agnostic', con:'slower (n_repeats \u00d7 fit)', c:C.blue},
# #   ].forEach(function(r){
# #     out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
# #       +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
# #       +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.pro+'</div>'
# #       +'<div style="font-size:8px;color:'+C.red+';margin-top:1px;">\u2718 '+r.con+'</div>'
# #       +'</div>';
# #   });
# #   out+='</div>';
# #   out+='</div></div>';
# #
# #   out+=insight('&#9888;&#65039;','MDI Bias Caveat',
# #     'MDI is <span style="color:'+C.red+';font-weight:700;">biased toward features with many unique values</span> — '
# #     +'they have more threshold candidates, increasing their chance of a high gain by chance. '
# #     +'For unbiased importance use <span style="color:'+C.accent+';font-family:monospace;">sklearn.inspection.permutation_importance</span> '
# #     +'or consider <span style="color:'+C.yellow+';font-weight:700;">SHAP values</span> for both direction and magnitude of influence.'
# #   );
# #   return out;
# # }
# #
# # /* ══════════════════════════════════════════════════════
# #    ROOT RENDER
# # ══════════════════════════════════════════════════════ */
# # var TABS=[
# #   '\u26d4 Intro &amp; Splits',
# #   '\u{1F4CA} Impurity Criteria',
# #   '\u{1F332} CART Growing',
# #   '\u2702\uFE0F Overfitting &amp; Pruning',
# #   '\u{1F4A1} Feature Importance'
# # ];
# #
# # function renderApp(){
# #   var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
# #   html+='<div style="text-align:center;margin-bottom:16px;">'
# #     +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.yellow+','+C.orange+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Decision Tree</div>'
# #     +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough \u2014 from axis-aligned splits and impurity to CART, pruning and feature importance')
# #     +'</div>';
# #   html+='<div class="tab-bar">';
# #   TABS.forEach(function(t,i){
# #     html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
# #   });
# #   html+='</div>';
# #   html+='<div class="fade">';
# #   if(S.tab===0)      html+=renderIntro();
# #   else if(S.tab===1) html+=renderImpurity();
# #   else if(S.tab===2) html+=renderCART();
# #   else if(S.tab===3) html+=renderOverfit();
# #   else if(S.tab===4) html+=renderFeatureImportance();
# #   html+='</div></div>';
# #   return html;
# # }
# #
# # function render(){
# #   document.getElementById('app').innerHTML=renderApp();
# #   bindEvents();
# # }
# #
# # function bindEvents(){
# #   document.querySelectorAll('[data-action]').forEach(function(el){
# #     var action=el.getAttribute('data-action');
# #     var idx=parseInt(el.getAttribute('data-idx'));
# #     var tag=el.tagName.toLowerCase();
# #     if(tag==='button'){
# #       el.addEventListener('click',function(){
# #         if(action==='tab')        {S.tab=idx; render();}
# #         else if(action==='cartStep'){S.cartStep=idx; render();}
# #         else if(action==='cartNext'){if(S.cartStep<4){S.cartStep++;render();}}
# #         else if(action==='cartPrev'){if(S.cartStep>0){S.cartStep--;render();}}
# #         else if(action==='featData'){S.featDataset=idx; render();}
# #       });
# #     } else if(tag==='input'){
# #       el.addEventListener('input',function(){
# #         var val=parseFloat(this.value);
# #         if(action==='dtDepth')      {S.dtDepth=Math.round(val);  render();}
# #         else if(action==='pVal')    {S.pVal=val;                 render();}
# #         else if(action==='treeDepth'){S.treeDepth=Math.round(val);render();}
# #         else if(action==='pruneAlpha'){S.pruneAlpha=val;         render();}
# #       });
# #     }
# #   });
# # }
# #
# # render();
# # </script>
# # </body>
# # </html>"""
# #
# # DT_VISUAL_HEIGHT = 1100