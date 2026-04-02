"""
Self-contained HTML visual for Random Forests.
5 interactive tabs: Ensemble Intro & Voting, Bootstrap Sampling,
Feature Subsampling, Variance Reduction, Aggregation & Prediction.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(RF_VISUAL_HTML, height=RF_VISUAL_HEIGHT, scrolling=True)
"""

RF_VISUAL_HTML = r"""<!DOCTYPE html>
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
  nTrees:1,
  bootTree:0,
  featTree:0,
  varN:10,
  aggMode:0
};

/* ══════════════════════════════════════════════════════
   TAB 0 — ENSEMBLE INTRO
══════════════════════════════════════════════════════ */
function rfAccuracy(n){return 0.92-0.22*Math.exp(-0.04*n);}
var ST_ACC=0.70;

function renderIntro(){
  var n=S.nTrees;
  var acc=rfAccuracy(n);
  var accCol=acc>0.88?C.green:acc>0.78?C.yellow:C.orange;

  /* All trees cycle through this 7-vote pattern */
  var votePattern=[1,0,1,1,0,1,1];
  var class1=0,class0=0;
  for(var i=0;i<n;i++){if(votePattern[i%7]===1)class1++;else class0++;}
  var winner=class1>=class0?1:0;
  var winnerCol=winner===1?C.accent:C.orange;
  var winnerLbl=winner===1?'Class A':'Class B';

  var svW=520,svH=290;
  var sv='';

  /* Title row */
  sv+='<text x="260" y="16" text-anchor="middle" fill="'+C.muted+'" font-size="10" font-family="monospace">'
    +n+' tree'+(n===1?'':'s')+' each vote \u2192 majority wins</text>';

  /* Draw up to 7 tree icons */
  var visN=Math.min(n,7);
  var treeW=46,gap=8;
  var totalW=visN*treeW+(visN-1)*gap;
  var startX=Math.max(10, Math.floor((360-totalW)/2));

  for(var i=0;i<visN;i++){
    var tx=startX+i*(treeW+gap);
    var ty=26;
    var vote=votePattern[i%7];
    var vc=vote===1?C.accent:C.orange;
    /* trunk */
    sv+='<rect x="'+(tx+treeW/2-3)+'" y="'+(ty+42)+'" width="6" height="10" rx="1" fill="'+hex(C.muted,0.4)+'"/>';
    /* canopy layers */
    sv+='<polygon points="'+(tx+treeW/2)+','+(ty+6)+' '+(tx+4)+','+(ty+26)+' '+(tx+treeW-4)+','+(ty+26)+'" fill="'+hex(vc,0.7)+'" stroke="'+vc+'" stroke-width="1"/>';
    sv+='<polygon points="'+(tx+treeW/2)+','+(ty+16)+' '+(tx+2)+','+(ty+40)+' '+(tx+treeW-2)+','+(ty+40)+'" fill="'+hex(vc,0.45)+'" stroke="'+vc+'" stroke-width="1"/>';
    /* vote badge */
    sv+='<text x="'+(tx+treeW/2)+'" y="'+(ty+63)+'" text-anchor="middle" fill="'+vc+'" font-size="11" font-family="monospace" font-weight="700">'+(vote===1?'A':'B')+'</text>';
    sv+='<text x="'+(tx+treeW/2)+'" y="'+(ty+74)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">T'+(i+1)+'</text>';
    /* arrow down */
    sv+='<line x1="'+(tx+treeW/2)+'" y1="'+(ty+76)+'" x2="'+(tx+treeW/2)+'" y2="108" stroke="'+C.dim+'" stroke-width="1" stroke-dasharray="3,2"/>';
    sv+='<polygon points="'+(tx+treeW/2-4)+',108 '+(tx+treeW/2)+',114 '+(tx+treeW/2+4)+',108" fill="'+C.dim+'"/>';
  }
  if(n>7){
    sv+='<text x="375" y="78" fill="'+C.dim+'" font-size="10" font-family="monospace">... +'+(n-7)+' more trees</text>';
  }

  /* Aggregation funnel */
  sv+='<polygon points="10,118 490,118 380,145 128,145" fill="'+hex(C.dim,0.15)+'" stroke="'+C.dim+'" stroke-width="1"/>';
  sv+='<text x="250" y="138" text-anchor="middle" fill="'+C.dim+'" font-size="9" font-family="monospace">AGGREGATE</text>';

  /* Final prediction box */
  sv+='<rect x="165" y="152" width="170" height="48" rx="8" fill="'+hex(winnerCol,0.15)+'" stroke="'+winnerCol+'" stroke-width="2"/>';
  sv+='<text x="250" y="169" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">MAJORITY VOTE</text>';
  sv+='<text x="250" y="188" text-anchor="middle" fill="'+winnerCol+'" font-size="14" font-family="monospace" font-weight="700">'+winnerLbl+'</text>';
  sv+='<text x="250" y="198" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+class1+'A vs '+class0+'B</text>';

  /* Vote distribution bar */
  var bY=218,bW=430,bH=22,bX=44;
  var frac=(n>0?class1/n:0);
  sv+='<text x="'+bX+'" y="'+(bY-5)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Vote distribution:</text>';
  sv+='<rect x="'+bX+'" y="'+bY+'" width="'+bW+'" height="'+bH+'" rx="4" fill="'+hex(C.orange,0.2)+'"/>';
  sv+='<rect x="'+bX+'" y="'+bY+'" width="'+(frac*bW).toFixed(1)+'" height="'+bH+'" rx="4" fill="'+hex(C.accent,0.6)+'"/>';
  sv+='<text x="'+(bX+6)+'" y="'+(bY+14)+'" fill="#0a0a0f" font-size="10" font-family="monospace" font-weight="700">A: '+class1+'</text>';
  sv+='<text x="'+(bX+bW-6)+'" y="'+(bY+14)+'" text-anchor="end" fill="'+C.orange+'" font-size="10" font-family="monospace" font-weight="700">B: '+class0+'</text>';

  /* Accuracy comparison bars */
  var cY=258,cW=380,cX=44;
  sv+='<text x="'+cX+'" y="'+(cY-4)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Accuracy vs single tree:</text>';
  sv+='<rect x="'+cX+'" y="'+cY+'" width="'+cW+'" height="18" rx="3" fill="'+hex(C.dim,0.3)+'"/>';
  sv+='<rect x="'+cX+'" y="'+cY+'" width="'+(ST_ACC*cW).toFixed(1)+'" height="18" rx="3" fill="'+hex(C.orange,0.45)+'"/>';
  sv+='<text x="'+(cX+5)+'" y="'+(cY+12)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Single Tree: '+(ST_ACC*100).toFixed(0)+'%</text>';
  sv+='<rect x="'+cX+'" y="'+(cY+22)+'" width="'+cW+'" height="18" rx="3" fill="'+hex(C.dim,0.3)+'"/>';
  sv+='<rect x="'+cX+'" y="'+(cY+22)+'" width="'+(acc*cW).toFixed(1)+'" height="18" rx="3" fill="'+hex(C.accent,0.55)+'"/>';
  sv+='<text x="'+(cX+5)+'" y="'+(cY+34)+'" fill="'+C.text+'" font-size="9" font-family="monospace">RF ('+n+' trees): '+(acc*100).toFixed(1)+'%</text>';

  var out=sectionTitle('Random Forest — Wisdom of the Crowd',
    'Train hundreds of diverse trees on randomised data, then aggregate their votes');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Ensemble Voting ('+n+' tree'+(n===1?'':'s')+')')
    +svgBox(sv,svW,svH)
    +sliderRow('nTrees',n,1,200,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.orange+';">\u2190 1 tree (high variance)</span>'
    +'<span style="color:'+C.green+';">200 trees (stable) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT FOREST');
  out+=statRow('n_estimators',n,C.accent);
  out+=statRow('Class A votes',class1,C.accent);
  out+=statRow('Class B votes',class0,C.orange);
  out+=statRow('Winner',winnerLbl,winnerCol);
  out+=statRow('RF accuracy',(acc*100).toFixed(1)+'%',accCol);
  out+=statRow('Single tree',(ST_ACC*100).toFixed(0)+'%',C.orange);
  out+=statRow('Improvement','+'+((acc-ST_ACC)*100).toFixed(1)+'%',C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TWO SOURCES OF RANDOMNESS');
  [{c:C.accent,lbl:'Bootstrap sampling',desc:'each tree sees ~63% unique rows'},
   {c:C.yellow,lbl:'Feature subsampling',desc:'each split considers only \u221ap features'}
  ].forEach(function(r){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VARIANCE FORMULA');
  out+=div('font-size:8.5px;color:'+C.muted+';line-height:2;font-family:monospace;',
    'Var(avg) = <span style="color:'+C.yellow+';">\u03c1</span>\u03c3\u00b2 + (1\u2212<span style="color:'+C.yellow+';">\u03c1</span>)\u03c3\u00b2/<span style="color:'+C.accent+';">B</span><br>'
    +'\u03c1 = inter-tree correlation<br>'
    +'B = n_estimators<br>'
    +'As B\u2192\u221e: Var \u2192 <span style="color:'+C.accent+';">\u03c1\u03c3\u00b2</span> (floor)'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127775;','Why More Trees Never Hurts',
    'Unlike increasing depth in a single tree, adding more trees <span style="color:'+C.green+';font-weight:700;">never causes overfitting</span>. '
    +'Each tree is trained on a different bootstrap sample with different random feature subsets. '
    +'More trees only reduce <span style="color:'+C.accent+';font-weight:700;">variance</span> — bias stays constant. '
    +'The only cost is compute time. Diminishing returns kick in after ~100\u2013200 trees.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 1 — BOOTSTRAP SAMPLING
══════════════════════════════════════════════════════ */
/* 3 precomputed bootstrap samples over n=10 rows */
var BOOT=[
  {name:'Tree 1',sample:[0,0,2,3,5,5,6,7,8,9],col:C.accent},
  {name:'Tree 2',sample:[1,2,3,4,4,5,6,8,9,9],col:C.yellow},
  {name:'Tree 3',sample:[0,1,2,3,6,7,7,8,9,9],col:C.purple}
];
var ROW_CLS=['A','B','A','A','B','A','B','A','B','A'];

function renderBootstrap(){
  var bt=S.bootTree;
  var bd=BOOT[bt];
  var sample=bd.sample;
  var col=bd.col;

  /* count how many times each row was drawn */
  var counts=new Array(10).fill(0);
  sample.forEach(function(i){counts[i]++;});
  var nUnique=counts.filter(function(c){return c>0;}).length;
  var nOOB=counts.filter(function(c){return c===0;}).length;

  var svW=520,svH=252;
  var sv='';

  var rW=43,rH=28,rGap=5;
  var rowStart=10;

  /* ── ROW 1: original dataset ── */
  sv+='<text x="'+rowStart+'" y="16" fill="'+C.muted+'" font-size="10" font-family="monospace">Original dataset (n = 10)</text>';
  for(var i=0;i<10;i++){
    var rx=rowStart+i*(rW+rGap);
    var isOOB=counts[i]===0;
    var rc=ROW_CLS[i]==='A'?C.blue:C.orange;
    var fillC=isOOB?C.dim:rc;
    var alpha=isOOB?0.8:0.25;
    sv+='<rect x="'+rx+'" y="22" width="'+rW+'" height="'+rH+'" rx="4"'
      +' fill="'+hex(fillC,alpha)+'" stroke="'+fillC+'" stroke-width="'+(isOOB?1:1.5)+'"/>';
    sv+='<text x="'+(rx+rW/2)+'" y="34" text-anchor="middle" fill="'+(isOOB?C.dim:fillC)+'" font-size="10" font-family="monospace" font-weight="700">R'+i+'</text>';
    sv+='<text x="'+(rx+rW/2)+'" y="46" text-anchor="middle" fill="'+(isOOB?C.dim:C.muted)+'" font-size="9" font-family="monospace">'+(isOOB?'OOB':ROW_CLS[i])+'</text>';
  }

  /* OOB legend marker */
  sv+='<rect x="'+rowStart+'" y="58" width="12" height="12" rx="2" fill="'+hex(C.dim,0.8)+'" stroke="'+C.dim+'" stroke-width="1"/>';
  sv+='<text x="'+(rowStart+16)+'" y="68" fill="'+C.muted+'" font-size="9" font-family="monospace">Out-of-Bag (OOB) — not seen by '+bd.name+'</text>';

  /* ── Arrow ── */
  sv+='<line x1="260" y1="80" x2="260" y2="98" stroke="'+col+'" stroke-width="1.5" stroke-dasharray="4,2"/>';
  sv+='<polygon points="255,98 260,105 265,98" fill="'+col+'"/>';
  sv+='<text x="270" y="95" fill="'+col+'" font-size="9" font-family="monospace">draw n=10 with replacement</text>';

  /* ── ROW 2: bootstrap sample ── */
  sv+='<text x="'+rowStart+'" y="120" fill="'+col+'" font-size="10" font-family="monospace" font-weight="700">'+bd.name+' bootstrap sample</text>';
  for(var i=0;i<10;i++){
    var rx=rowStart+i*(rW+rGap);
    var srcIdx=sample[i];
    var rc=ROW_CLS[srcIdx]==='A'?C.blue:C.orange;
    sv+='<rect x="'+rx+'" y="126" width="'+rW+'" height="'+rH+'" rx="4"'
      +' fill="'+hex(rc,0.6)+'" stroke="'+rc+'" stroke-width="1.5"/>';
    sv+='<text x="'+(rx+rW/2)+'" y="138" text-anchor="middle" fill="#0a0a0f" font-size="10" font-family="monospace" font-weight="700">R'+srcIdx+'</text>';
    sv+='<text x="'+(rx+rW/2)+'" y="150" text-anchor="middle" fill="#0a0a0f" font-size="9" font-family="monospace">'+ROW_CLS[srcIdx]+'</text>';
    /* duplicate badge */
    if(counts[srcIdx]>1&&sample.indexOf(srcIdx)===i){
      sv+='<circle cx="'+(rx+rW-5)+'" cy="130" r="7" fill="'+col+'" stroke="#0a0a0f" stroke-width="1"/>';
      sv+='<text x="'+(rx+rW-5)+'" y="134" text-anchor="middle" fill="#0a0a0f" font-size="8" font-family="monospace" font-weight="700">\u00d7'+counts[srcIdx]+'</text>';
    }
  }

  /* ── Legend ── */
  sv+='<rect x="'+rowStart+'" y="170" width="12" height="12" rx="2" fill="'+hex(C.blue,0.25)+'" stroke="'+C.blue+'" stroke-width="1.5"/>';
  sv+='<text x="'+(rowStart+16)+'" y="180" fill="'+C.muted+'" font-size="9" font-family="monospace">Class A</text>';
  sv+='<rect x="88" y="170" width="12" height="12" rx="2" fill="'+hex(C.orange,0.25)+'" stroke="'+C.orange+'" stroke-width="1.5"/>';
  sv+='<text x="104" y="180" fill="'+C.muted+'" font-size="9" font-family="monospace">Class B</text>';
  sv+='<circle cx="168" cy="176" r="6" fill="'+col+'" stroke="#0a0a0f" stroke-width="1"/>';
  sv+='<text x="178" y="180" fill="'+C.muted+'" font-size="9" font-family="monospace">\u00d7N = row sampled N times</text>';

  /* ── Poisson 63.2% math ── */
  sv+='<text x="'+rowStart+'" y="208" fill="'+C.muted+'" font-size="9" font-family="monospace">P(row excluded) = (1\u22121/n)\u207f  \u2192  e\u207b\u00b9 \u2248 36.8%    \u2234 unique rows \u2248 63.2%</text>';

  /* OOB arrow indicator for each OOB row */
  for(var i=0;i<10;i++){
    if(counts[i]===0){
      var rx=rowStart+i*(rW+rGap);
      sv+='<text x="'+(rx+rW/2)+'" y="80" text-anchor="middle" fill="'+C.red+'" font-size="10">OOB</text>';
    }
  }

  var out=sectionTitle('Bootstrap Sampling','Each tree trains on a random subset of rows drawn with replacement from the full training set');

  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;">';
  BOOT.forEach(function(b,i){
    out+=btnSel(i,bt,b.col,'\u{1F332} '+b.name,'bootTree');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+col+';margin-bottom:10px;',bd.name+' — Bootstrap Sample')
    +svgBox(sv,svW,svH)
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','SAMPLE STATISTICS');
  out+=statRow('Dataset size (n)',10,C.text);
  out+=statRow('Bootstrap size',10,C.text);
  out+=statRow('Unique rows selected',nUnique+' / 10',col);
  out+=statRow('% unique rows',(nUnique/10*100).toFixed(0)+'%',col);
  out+=statRow('OOB rows (not selected)',nOOB+' / 10',C.red);
  out+=statRow('OOB percentage',(nOOB/10*100).toFixed(0)+'%',C.red);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE 63.2% RULE');
  out+=div('font-size:8.5px;color:'+C.muted+';line-height:2;font-family:monospace;background:#08080d;padding:8px;border-radius:6px;border:1px solid '+C.border+';',
    'P(row i excluded from sample):<br>'
    +'  (1 \u2212 1/n)\u207f\u2009\u27f6\u2009e\u207b\u00b9 \u2248 <span style="color:'+C.red+';font-weight:700;">36.8%</span><br><br>'
    +'Each bootstrap sample contains<br>'
    +'~<span style="color:'+col+';font-weight:700;">63.2%</span> unique training rows\n'
    +'Remaining ~<span style="color:'+C.red+';font-weight:700;">36.8%</span> = OOB set'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','OOB AS FREE VALIDATION');
  out+=div('font-size:8.5px;color:'+C.muted+';line-height:1.8;',
    'Each row is OOB for ~37% of trees. Those trees never saw it during training — '
    +'so their vote is <span style="color:'+C.green+';font-weight:700;">unbiased</span>. '
    +'Aggregating OOB votes gives the <span style="color:'+C.accent+';font-weight:700;">OOB error</span>: '
    +'a free generalisation estimate without a separate validation set.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128218;','OOB Error \u2248 3-Fold Cross-Validation',
    'Empirically, OOB error tracks 3\u20135 fold CV accuracy very closely. '
    +'Enable with <span style="color:'+C.accent+';font-family:monospace;">oob_score=True</span> in sklearn. '
    +'Because no data is withheld entirely, OOB is especially useful on small datasets '
    +'where every training example matters.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 2 — FEATURE SUBSAMPLING
══════════════════════════════════════════════════════ */
var FEAT_NAMES=['Age','Income','Credit','Employed','Debt','Location'];
var FEAT_COLS=[C.accent,C.yellow,C.blue,C.green,C.purple,C.orange];

/* For each tree, 3 splits: which 2 features were considered, which won */
var FEAT_SEL=[
  [[0,1],[2,5],[3,4]],  /* Tree 1 splits */
  [[1,3],[0,4],[2,5]],  /* Tree 2 splits */
  [[2,4],[1,5],[0,3]]   /* Tree 3 splits */
];
var FEAT_BEST=[
  [1,2,3],  /* Tree 1 best features per split */
  [3,4,5],  /* Tree 2 */
  [4,1,0]   /* Tree 3 */
];

function renderFeatureSubsampling(){
  var ft=S.featTree;
  var treeCol=[C.accent,C.yellow,C.purple][ft];
  var allSplits=FEAT_SEL[ft];
  var bestFs=FEAT_BEST[ft];
  var p=6,sqrtP=2;

  var svW=520,svH=310;
  var sv='';

  /* ── Feature circles at top ── */
  var fR=22,fGap=72,fStartX=28,fY=50;
  sv+='<text x="'+fStartX+'" y="20" fill="'+C.muted+'" font-size="10" font-family="monospace">All p=6 features available:</text>';
  FEAT_NAMES.forEach(function(f,i){
    var fx=fStartX+i*fGap;
    /* check if this feature was ever used by any split */
    var anyUsed=allSplits.some(function(s){return s.indexOf(i)>=0;});
    var isBest=bestFs.indexOf(i)>=0;
    var fc=anyUsed?FEAT_COLS[i]:C.dim;
    var fa=isBest?0.85:anyUsed?0.35:0.12;
    var bw=isBest?2.5:anyUsed?1.8:1;
    sv+='<circle cx="'+fx+'" cy="'+fY+'" r="'+fR+'" fill="'+hex(fc,fa)+'" stroke="'+fc+'" stroke-width="'+bw+'"/>';
    sv+='<text x="'+fx+'" y="'+(fY+4)+'" text-anchor="middle" fill="'+(anyUsed?'#0a0a0f':C.dim)+'" font-size="8" font-family="monospace" font-weight="700">'+f+'</text>';
    sv+='<text x="'+fx+'" y="'+(fY+fR+12)+'" text-anchor="middle" fill="'+fc+'" font-size="8" font-family="monospace">'+(isBest?'\u2605best':anyUsed?'tried':'skip')+'</text>';
  });

  /* ── 3 split nodes ── */
  var nX=[80,240,400],nY=110,nW=110,nH=38;
  allSplits.forEach(function(sel,si){
    var nx=nX[si],ny=nY;
    var bf=bestFs[si];
    var bc=FEAT_COLS[bf];
    sv+='<rect x="'+nx+'" y="'+ny+'" width="'+nW+'" height="'+nH+'" rx="6"'
      +' fill="'+hex(treeCol,0.1)+'" stroke="'+treeCol+'" stroke-width="1.5"/>';
    sv+='<text x="'+(nx+nW/2)+'" y="'+(ny+14)+'" text-anchor="middle" fill="'+treeCol+'" font-size="9" font-family="monospace" font-weight="700">Split '+(si+1)+'</text>';
    sv+='<text x="'+(nx+nW/2)+'" y="'+(ny+26)+'" text-anchor="middle" fill="'+C.text+'" font-size="9" font-family="monospace">\u2605 '+FEAT_NAMES[bf]+'</text>';
    sv+='<text x="'+(nx+nW/2)+'" y="'+(ny+37)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">from '+sqrtP+' of '+p+'</text>';
    /* dashed lines from feature circles down to split node */
    sel.forEach(function(fi){
      var fx=fStartX+fi*fGap;
      var fy=fY+fR;
      var tx2=nx+nW/2;
      var ty2=ny;
      sv+='<line x1="'+fx+'" y1="'+fy+'" x2="'+tx2.toFixed(1)+'" y2="'+ty2+'"'
        +' stroke="'+hex(FEAT_COLS[fi],0.45)+'" stroke-width="1.5" stroke-dasharray="4,2"/>';
    });
  });

  /* ── Feature candidate rows ── */
  sv+='<text x="10" y="168" fill="'+C.muted+'" font-size="9" font-family="monospace">Features considered per split (Tree '+[1,2,3][ft]+'):</text>';
  allSplits.forEach(function(sel,si){
    var ry=178+si*30;
    sv+='<text x="10" y="'+(ry+11)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">Split '+(si+1)+':</text>';
    sel.forEach(function(fi,k){
      var bx=60+k*120;
      var isBest=bestFs[si]===fi;
      var fc=FEAT_COLS[fi];
      sv+='<rect x="'+bx+'" y="'+ry+'" width="112" height="20" rx="4"'
        +' fill="'+hex(fc,isBest?0.55:0.15)+'" stroke="'+fc+'" stroke-width="'+(isBest?2:1)+'"/>';
      sv+='<text x="'+(bx+56)+'" y="'+(ry+13)+'" text-anchor="middle"'
        +' fill="'+(isBest?'#0a0a0f':fc)+'" font-size="9" font-family="monospace" font-weight="'+(isBest?'700':'400')+'">'+FEAT_NAMES[fi]+(isBest?' \u2605':'')+'</text>';
    });
  });

  /* ── Decorrelation illustration ── */
  var dY=272;
  sv+='<text x="10" y="'+dY+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Root features chosen by each tree:</text>';
  [[0,C.accent,'T1: Income'],[1,C.yellow,'T2: Employed'],[2,C.purple,'T3: Debt']].forEach(function(t,i){
    var bx=10+i*160;
    sv+='<rect x="'+bx+'" y="'+(dY+8)+'" width="152" height="20" rx="4"'
      +' fill="'+hex(t[1],0.2)+'" stroke="'+t[1]+'" stroke-width="1.5"/>';
    sv+='<text x="'+(bx+76)+'" y="'+(dY+21)+'" text-anchor="middle" fill="'+t[1]+'" font-size="9" font-family="monospace" font-weight="700">'+t[2]+'</text>';
  });

  var out=sectionTitle('Feature Subsampling at Each Split',
    'At every node, only \u221ap random features are candidates — not all p features');

  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;">';
  ['Tree 1','Tree 2','Tree 3'].forEach(function(t,i){
    out+=btnSel(i,ft,[C.accent,C.yellow,C.purple][i],'\u{1F332} '+t,'featTree');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+treeCol+';margin-bottom:10px;','Tree '+[1,2,3][ft]+' — Feature Candidates per Split')
    +svgBox(sv,svW,svH)
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','MAX_FEATURES SETTINGS');
  [{v:'sqrt(p)',desc:'classification default — \u221a6 \u2248 2',c:C.accent,note:'recommended'},
   {v:'log2(p)',desc:'alternative for large p',c:C.yellow,note:'alternative'},
   {v:'0.33\u00b7p',desc:'regression default',c:C.blue,note:'regression'},
   {v:'None',desc:'all features \u2192 no randomness',c:C.red,note:'no diversity'}
  ].forEach(function(r){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="display:flex;justify-content:space-between;">'
      +'<span style="font-size:9px;font-family:monospace;font-weight:700;color:'+r.c+';">'+r.v+'</span>'
      +'<span style="font-size:8px;color:'+C.dim+';">'+r.note+'</span></div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY SUBSAMPLE FEATURES?');
  out+=div('font-size:8.5px;color:'+C.muted+';line-height:1.8;',
    'Without feature subsampling, every tree would always pick the same '
    +'<span style="color:'+C.yellow+';font-weight:700;">strongest root feature</span> — '
    +'producing highly correlated trees whose errors do <em>not</em> cancel. '
    +'Forcing random subsets creates <span style="color:'+C.accent+';font-weight:700;">tree diversity</span>, '
    +'lowering \u03c1 in the variance formula so ensemble averaging becomes meaningful.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127381;','max_features — The Key Hyperparameter',
    'max_features is the most important knob in a Random Forest. '
    +'Smaller values create <span style="color:'+C.green+';font-weight:700;">more diverse trees</span> (lower correlation \u03c1) '
    +'but increase <span style="color:'+C.red+';font-weight:700;">individual tree bias</span>. '
    +'sqrt(p) sits at the sweet spot for classification. '
    +'The variance formula Var = \u03c1\u03c3\u00b2 + (1\u2212\u03c1)\u03c3\u00b2/B shows that reducing \u03c1 '
    +'is just as powerful as increasing B.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 3 — VARIANCE REDUCTION
══════════════════════════════════════════════════════ */
function oobErr(n){return 0.09+0.19*Math.exp(-0.038*n);}
var ST_ERR=0.28;

function renderVarianceReduction(){
  var n=S.varN;
  var oob=oobErr(n);
  var oobCol=oob<0.12?C.green:oob<0.18?C.yellow:C.orange;
  var improve=((ST_ERR-oob)/ST_ERR*100).toFixed(0);

  var sv=plotAxes('n_estimators (trees)','OOB / Test Error',200,0.5,
    [0,50,100,150,200],[0,0.1,0.2,0.3,0.4,0.5]);

  /* single tree baseline */
  sv+='<line x1="'+PL+'" y1="'+sy(ST_ERR,0.5).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(ST_ERR,0.5).toFixed(1)+'"'
    +' stroke="'+C.orange+'" stroke-width="1.5" stroke-dasharray="6,3"/>';
  sv+='<rect x="'+(PL+PW-130)+'" y="'+(sy(ST_ERR,0.5)-14).toFixed(1)+'" width="128" height="13" rx="3" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(PL+PW-6)+'" y="'+(sy(ST_ERR,0.5)-4).toFixed(1)+'" text-anchor="end"'
    +' fill="'+C.orange+'" font-size="10" font-family="monospace">Single Tree: '+(ST_ERR*100).toFixed(0)+'%</text>';

  /* RF OOB error curve */
  var rfPath='';
  for(var k=1;k<=200;k++){
    var e=oobErr(k);
    rfPath+=(k===1?'M':'L')+sx(k,200).toFixed(1)+','+sy(e,0.5).toFixed(1)+' ';
  }
  sv+='<path d="'+rfPath+'" fill="none" stroke="'+C.accent+'" stroke-width="2.5"/>';

  /* asymptote */
  var asym=oobErr(200);
  sv+='<line x1="'+PL+'" y1="'+sy(asym,0.5).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(asym,0.5).toFixed(1)+'"'
    +' stroke="'+hex(C.accent,0.3)+'" stroke-width="1" stroke-dasharray="3,4"/>';
  sv+='<text x="'+(PL+4)+'" y="'+(sy(asym,0.5)-4).toFixed(1)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">\u03c1\u03c3\u00b2 floor</text>';

  /* current n vertical */
  var cx2=sx(n,200),cy2=sy(oob,0.5);
  sv+='<line x1="'+cx2.toFixed(1)+'" y1="'+PT+'" x2="'+cx2.toFixed(1)+'" y2="'+(PT+PH)+'"'
    +' stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cx2.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" r="5.5" fill="'+oobCol+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  var lbX=Math.min(cx2+8,PL+PW-80);
  sv+='<rect x="'+lbX.toFixed(1)+'" y="'+(cy2-14).toFixed(1)+'" width="78" height="13" rx="3" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="'+(lbX+4).toFixed(1)+'" y="'+(cy2-4).toFixed(1)+'" fill="'+oobCol+'" font-size="10" font-family="monospace" font-weight="700">err='+(oob*100).toFixed(1)+'%</text>';

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+PH-36)+'" width="180" height="34" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-22)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-22)+'" stroke="'+C.accent+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-19)+'" fill="'+C.accent+'" font-size="10" font-family="monospace">RF OOB error</text>';
  sv+='<line x1="'+(PL+10)+'" y1="'+(PT+PH-10)+'" x2="'+(PL+30)+'" y2="'+(PT+PH-10)+'" stroke="'+C.orange+'" stroke-width="1.5" stroke-dasharray="6,3"/>';
  sv+='<text x="'+(PL+34)+'" y="'+(PT+PH-7)+'" fill="'+C.orange+'" font-size="10" font-family="monospace">Single tree baseline</text>';

  var out=sectionTitle('Variance Reduction',
    'More trees reduce prediction variance — test error drops sharply then hits a floor at \u03c1\u03c3\u00b2');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','OOB Error vs n_estimators')
    +svgBox(sv)
    +sliderRow('varN',n,1,200,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.orange+';">\u2190 1 tree (high variance)</span>'
    +'<span style="color:'+C.green+';">200 trees (stable) \u2192</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT METRICS');
  out+=statRow('n_estimators',n,C.accent);
  out+=statRow('OOB error',(oob*100).toFixed(1)+'%',oobCol);
  out+=statRow('Single tree error',(ST_ERR*100).toFixed(0)+'%',C.orange);
  out+=statRow('Error reduction',improve+'%',C.green);
  out+=statRow('Asymptote (n\u2192\u221e)',(asym*100).toFixed(1)+'%',C.muted);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VARIANCE DECOMPOSITION');
  out+=div('font-size:8.5px;color:'+C.muted+';line-height:2.2;font-family:monospace;background:#08080d;padding:8px;border-radius:6px;border:1px solid '+C.border+';',
    'Var(ensemble) =<br>'
    +'  <span style="color:'+C.yellow+';">\u03c1</span>\u03c3\u00b2 + (1\u2212<span style="color:'+C.yellow+';">\u03c1</span>)\u03c3\u00b2/<span style="color:'+C.accent+';">B</span><br>'
    +'\u03c1 = inter-tree correlation<br>'
    +'\u03c3\u00b2 = single tree variance<br>'
    +'<span style="color:'+C.accent+';">B</span> = n_estimators<br>'
    +'B\u2192\u221e: floor = <span style="color:'+C.yellow+';">\u03c1</span>\u03c3\u00b2'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PRACTICAL RULES');
  [{c:C.green, lbl:'100\u2013200 trees captures 95%+ of gains'},
   {c:C.yellow,lbl:'More trees never hurts accuracy'},
   {c:C.blue,  lbl:'Speed up: n_jobs=-1 (parallel fit)'},
   {c:C.orange,lbl:'Diminishing returns after ~100 trees'}
  ].forEach(function(r){
    out+='<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;">'
      +'<div style="width:8px;height:8px;border-radius:50%;background:'+r.c+';flex-shrink:0;"></div>'
      +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','Bias is Untouched — Only Variance Falls',
    'Averaging B trees with bias \u03b2 gives an ensemble with the <span style="color:'+C.red+';font-weight:700;">same bias \u03b2</span>. '
    +'Variance falls proportionally to 1/B down to the correlation floor \u03c1\u03c3\u00b2. '
    +'This is why Random Forests use <span style="color:'+C.accent+';font-weight:700;">deep trees (low bias)</span> as base learners. '
    +'If trees are too shallow, the ensemble just averages high-bias models — averaging cannot fix bias.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 4 — AGGREGATION & PREDICTION
══════════════════════════════════════════════════════ */
var CLASSIF_VOTES=[
  {vote:'A',prob:0.82,col:C.accent},
  {vote:'B',prob:0.61,col:C.orange},
  {vote:'A',prob:0.77,col:C.accent},
  {vote:'A',prob:0.55,col:C.accent},
  {vote:'B',prob:0.70,col:C.orange}
];
var REGR_PREDS=[3.2,4.1,3.8,4.5,3.5];

function renderAggregation(){
  var mode=S.aggMode;
  var isC=mode===0;
  var svW=520,svH=275;
  var sv='';

  if(isC){
    /* ── Classification: majority vote ── */
    var votesA=CLASSIF_VOTES.filter(function(t){return t.vote==='A';}).length;
    var votesB=CLASSIF_VOTES.filter(function(t){return t.vote==='B';}).length;
    var winner=votesA>votesB?'A':'B';
    var wc=winner==='A'?C.accent:C.orange;

    /* Tree boxes */
    CLASSIF_VOTES.forEach(function(t,i){
      var tx=10+i*100,ty=18;
      var vc=t.vote==='A'?C.accent:C.orange;
      sv+='<rect x="'+tx+'" y="'+ty+'" width="88" height="80" rx="6" fill="'+hex(vc,0.1)+'" stroke="'+vc+'" stroke-width="1.5"/>';
      sv+='<text x="'+(tx+44)+'" y="'+(ty+14)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Tree '+(i+1)+'</text>';
      sv+='<polygon points="'+(tx+44)+','+(ty+22)+' '+(tx+28)+','+(ty+46)+' '+(tx+60)+','+(ty+46)+'" fill="'+hex(vc,0.55)+'" stroke="'+vc+'" stroke-width="1"/>';
      sv+='<rect x="'+(tx+41)+'" y="'+(ty+46)+'" width="6" height="9" rx="1" fill="'+hex(C.muted,0.35)+'"/>';
      sv+='<text x="'+(tx+44)+'" y="'+(ty+68)+'" text-anchor="middle" fill="'+vc+'" font-size="13" font-family="monospace" font-weight="700">'+t.vote+'</text>';
      sv+='<line x1="'+(tx+44)+'" y1="'+(ty+80)+'" x2="'+(tx+44)+'" y2="112" stroke="'+C.dim+'" stroke-width="1.5"/>';
      sv+='<polygon points="'+(tx+40)+',112 '+(tx+44)+',118 '+(tx+48)+',112" fill="'+C.dim+'"/>';
    });

    /* Aggregation funnel */
    sv+='<polygon points="10,122 500,122 388,155 120,155" fill="'+hex(C.dim,0.18)+'" stroke="'+C.dim+'" stroke-width="1"/>';
    sv+='<text x="255" y="143" text-anchor="middle" fill="'+C.dim+'" font-size="9" font-family="monospace">AGGREGATE \u2014 majority vote</text>';

    /* Winner box */
    sv+='<rect x="168" y="162" width="174" height="52" rx="8" fill="'+hex(wc,0.15)+'" stroke="'+wc+'" stroke-width="2"/>';
    sv+='<text x="255" y="179" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">MAJORITY VOTE</text>';
    sv+='<text x="255" y="198" text-anchor="middle" fill="'+wc+'" font-size="14" font-family="monospace" font-weight="700">Predict: Class '+winner+'</text>';
    sv+='<text x="255" y="210" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+votesA+'A vs '+votesB+'B  \u2014  conf: '+(Math.max(votesA,votesB)/5*100).toFixed(0)+'%</text>';

    /* Vote bar */
    var bX=44,bY=232,bW=430,bH=22;
    sv+='<rect x="'+bX+'" y="'+bY+'" width="'+bW+'" height="'+bH+'" rx="4" fill="'+hex(C.orange,0.2)+'"/>';
    sv+='<rect x="'+bX+'" y="'+bY+'" width="'+(votesA/5*bW).toFixed(1)+'" height="'+bH+'" rx="4" fill="'+hex(C.accent,0.6)+'"/>';
    sv+='<text x="'+(bX+6)+'" y="'+(bY+14)+'" fill="#0a0a0f" font-size="10" font-family="monospace" font-weight="700">A: '+votesA+'/5</text>';
    sv+='<text x="'+(bX+bW-6)+'" y="'+(bY+14)+'" text-anchor="end" fill="'+C.orange+'" font-size="10" font-family="monospace" font-weight="700">B: '+votesB+'/5</text>';

    /* probability note */
    sv+='<text x="255" y="270" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Soft proba: mean of tree probabilities \u2014 e.g. P(A) = '+((0.82+0.77+0.55)/5).toFixed(2)+' (3 trees with P(A) recorded)</text>';

  } else {
    /* ── Regression: mean ── */
    var preds=REGR_PREDS;
    var mean=preds.reduce(function(s,v){return s+v;},0)/preds.length;

    preds.forEach(function(p2,i){
      var tx=10+i*100,ty=18;
      sv+='<rect x="'+tx+'" y="'+ty+'" width="88" height="80" rx="6" fill="'+hex(C.blue,0.1)+'" stroke="'+C.blue+'" stroke-width="1.5"/>';
      sv+='<text x="'+(tx+44)+'" y="'+(ty+14)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Tree '+(i+1)+'</text>';
      sv+='<polygon points="'+(tx+44)+','+(ty+22)+' '+(tx+28)+','+(ty+46)+' '+(tx+60)+','+(ty+46)+'" fill="'+hex(C.blue,0.55)+'" stroke="'+C.blue+'" stroke-width="1"/>';
      sv+='<rect x="'+(tx+41)+'" y="'+(ty+46)+'" width="6" height="9" rx="1" fill="'+hex(C.muted,0.35)+'"/>';
      sv+='<text x="'+(tx+44)+'" y="'+(ty+68)+'" text-anchor="middle" fill="'+C.blue+'" font-size="12" font-family="monospace" font-weight="700">'+p2.toFixed(1)+'</text>';
      sv+='<line x1="'+(tx+44)+'" y1="'+(ty+80)+'" x2="'+(tx+44)+'" y2="112" stroke="'+C.dim+'" stroke-width="1.5"/>';
      sv+='<polygon points="'+(tx+40)+',112 '+(tx+44)+',118 '+(tx+48)+',112" fill="'+C.dim+'"/>';
    });

    sv+='<polygon points="10,122 500,122 388,155 120,155" fill="'+hex(C.dim,0.18)+'" stroke="'+C.dim+'" stroke-width="1"/>';
    sv+='<text x="255" y="143" text-anchor="middle" fill="'+C.dim+'" font-size="9" font-family="monospace">AGGREGATE \u2014 mean of outputs</text>';

    sv+='<rect x="168" y="162" width="174" height="52" rx="8" fill="'+hex(C.blue,0.15)+'" stroke="'+C.blue+'" stroke-width="2"/>';
    sv+='<text x="255" y="179" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">MEAN AVERAGE</text>';
    sv+='<text x="255" y="198" text-anchor="middle" fill="'+C.blue+'" font-size="14" font-family="monospace" font-weight="700">y\u0302 = '+mean.toFixed(2)+'</text>';
    sv+='<text x="255" y="210" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">mean of '+preds.length+' tree outputs</text>';

    /* number line */
    var lX1=44,lX2=474,lY=240,vMin=2,vMax=6;
    sv+='<line x1="'+lX1+'" y1="'+lY+'" x2="'+lX2+'" y2="'+lY+'" stroke="'+C.dim+'" stroke-width="2"/>';
    sv+='<text x="'+lX1+'" y="'+(lY+14)+'" text-anchor="middle" fill="'+C.dim+'" font-size="9" font-family="monospace">'+vMin+'</text>';
    sv+='<text x="'+lX2+'" y="'+(lY+14)+'" text-anchor="middle" fill="'+C.dim+'" font-size="9" font-family="monospace">'+vMax+'</text>';

    preds.forEach(function(p2){
      var px2=lX1+(p2-vMin)/(vMax-vMin)*(lX2-lX1);
      sv+='<circle cx="'+px2.toFixed(1)+'" cy="'+lY+'" r="5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1"/>';
      sv+='<text x="'+px2.toFixed(1)+'" y="'+(lY-8)+'" text-anchor="middle" fill="'+C.blue+'" font-size="8" font-family="monospace">'+p2.toFixed(1)+'</text>';
    });
    var mPx=lX1+(mean-vMin)/(vMax-vMin)*(lX2-lX1);
    sv+='<circle cx="'+mPx.toFixed(1)+'" cy="'+lY+'" r="8" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="2"/>';
    sv+='<text x="'+mPx.toFixed(1)+'" y="'+(lY+18)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="9" font-family="monospace" font-weight="700">y\u0302='+mean.toFixed(2)+'</text>';

    sv+='<text x="255" y="270" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Averaging reduces variance: Var(mean) = Var(single tree) / n_estimators (if trees uncorrelated)</text>';
  }

  var out=sectionTitle('Prediction Aggregation',
    'Classification uses majority vote; regression uses the mean of tree outputs');

  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;">';
  out+=btnSel(0,mode,C.accent,'\u{1F3F7}\uFE0F Classification (Majority Vote)','aggMode');
  out+=btnSel(1,mode,C.blue,'\u{1F4C9} Regression (Mean)','aggMode');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+(isC?C.accent:C.blue)+';margin-bottom:10px;',
        isC?'5-Tree Majority Vote':'5-Tree Mean Average')
    +svgBox(sv,svW,svH)
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','AGGREGATION METHODS');
  [{lbl:'Majority vote',desc:'most common class label across all trees',c:C.accent,when:'classification'},
   {lbl:'Soft vote (proba)',desc:'average per-class probability estimates',c:C.yellow,when:'classification'},
   {lbl:'Mean',desc:'average of all tree output values',c:C.blue,when:'regression'},
   {lbl:'Median',desc:'robust to outlier tree predictions',c:C.purple,when:'regression'}
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="display:flex;justify-content:space-between;">'
      +'<span style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</span>'
      +'<span style="font-size:8px;color:'+C.dim+';">'+r.when+'</span></div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY PROPERTIES');
  [{icon:'\u2714',lbl:'Embarrassingly parallel (n_jobs=-1)',c:C.green},
   {icon:'\u2714',lbl:'Handles mixed feature types',c:C.green},
   {icon:'\u2714',lbl:'OOB score = free validation',c:C.green},
   {icon:'\u2714',lbl:'Minimal preprocessing required',c:C.green},
   {icon:'\u2718',lbl:'Cannot extrapolate beyond training range',c:C.red},
   {icon:'\u26a0',lbl:'MDI importance biased to high-cardinality',c:C.yellow}
  ].forEach(function(r){
    out+='<div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;font-size:9px;">'
      +'<span style="color:'+r.c+';font-weight:700;flex-shrink:0;">'+r.icon+'</span>'
      +'<span style="color:'+C.muted+';">'+r.lbl+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','RF vs Gradient Boosting — When to Use Which',
    'Random Forests work <span style="color:'+C.green+';font-weight:700;">out of the box</span> with minimal tuning — '
    +'no feature scaling, handles mixed types, robust hyperparameters. '
    +'Random Forests reduce <span style="color:'+C.accent+';font-weight:700;">variance only</span> (bias = single deep tree). '
    +'Gradient Boosting also reduces <span style="color:'+C.yellow+';font-weight:700;">bias</span> by training trees on residuals sequentially — '
    +'making it more accurate but more sensitive to hyperparameters and prone to overfitting.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════ */
var TABS=[
  '&#127775; Ensemble Intro',
  '&#128218; Bootstrap Sampling',
  '&#127922; Feature Subsampling',
  '&#128200; Variance Reduction',
  '&#9889; Aggregation'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.yellow+','+C.orange+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Random Forest</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough \u2014 from ensemble voting and bootstrap to variance reduction and aggregation')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderBootstrap();
  else if(S.tab===2) html+=renderFeatureSubsampling();
  else if(S.tab===3) html+=renderVarianceReduction();
  else if(S.tab===4) html+=renderAggregation();
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
        if(action==='tab')         {S.tab=idx;       render();}
        else if(action==='bootTree'){S.bootTree=idx;  render();}
        else if(action==='featTree'){S.featTree=idx;  render();}
        else if(action==='aggMode') {S.aggMode=idx;   render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='nTrees')    {S.nTrees=Math.round(val); render();}
        else if(action==='varN') {S.varN=Math.round(val);   render();}
      });
    }
  });
}

render();

/* ─── CTRL+SCROLL ZOOM ─── */
var ZOOM = 1.0;
var zoomToast = null;

function applyZoom(){
  document.body.style.zoom = ZOOM;
  // show toast
  clearTimeout(zoomToast);
  var toast = document.getElementById('zoom-toast');
  if(!toast){
    toast = document.createElement('div');
    toast.id = 'zoom-toast';
    toast.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#12121a;border:1px solid #4ecdc4;'
      +'color:#4ecdc4;font-family:monospace;font-size:12px;font-weight:700;padding:8px 14px;'
      +'border-radius:8px;z-index:9999;pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(toast);
  }
  toast.textContent = 'zoom ' + Math.round(ZOOM*100) + '%';
  toast.style.opacity = '1';
  zoomToast = setTimeout(function(){ toast.style.opacity = '0'; }, 1200);
}

document.addEventListener('wheel', function(e){
  if(!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  var delta = e.deltaY > 0 ? -0.05 : 0.05;
  ZOOM = Math.min(3.0, Math.max(0.4, ZOOM + delta));
  // round to 2 decimals
  ZOOM = Math.round(ZOOM * 100) / 100;
  applyZoom();
}, {passive: false});
</script>
</body>
</html>"""

RF_VISUAL_HEIGHT = 1080