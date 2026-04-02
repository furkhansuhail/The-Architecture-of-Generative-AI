"""
Self-contained HTML visual for Support Vector Machines (SVM).
5 interactive tabs: Intro & Decision Boundary, Margin & Support Vectors,
Soft Margin (C Parameter), Kernel Trick, Multi-class SVM.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(SVM_VISUAL_HTML, height=SVM_VISUAL_HEIGHT, scrolling=True)
"""

SVM_VISUAL_HTML = r"""<!DOCTYPE html>
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

/* ─── DATA ─── */
var PTS_A=[
  {x:2.0,y:3.5},{x:1.5,y:6.0},{x:2.8,y:2.5},{x:3.2,y:5.2},
  {x:3.8,y:4.0},{x:2.5,y:7.5},{x:1.2,y:4.8},{x:3.5,y:2.8},
  {x:2.0,y:8.5},{x:4.0,y:6.5}
];
var PTS_B=[
  {x:7.0,y:4.2},{x:6.5,y:7.0},{x:8.2,y:3.0},{x:7.8,y:6.5},
  {x:8.5,y:5.2},{x:6.2,y:8.2},{x:7.5,y:2.8},{x:8.8,y:7.2},
  {x:6.0,y:5.5},{x:9.0,y:4.5}
];
/* optimal hyperplane x=5, support vectors at x=4 (A) and x=6 (B) */
var TRUE_W1=1,TRUE_W2=0,TRUE_B=-5;
var SUPPORT_A=[{x:4.0,y:4.0},{x:4.0,y:6.0}];
var SUPPORT_B=[{x:6.0,y:4.0},{x:6.0,y:6.0}];

/* non-linearly separable circular data for kernel tab */
var CIRC_IN=[], CIRC_OUT=[];
(function(){
  var cx=5,cy=5;
  var r1=1.7, r2=3.1;
  [0,36,72,108,144,180,216,252,288,324].forEach(function(a){
    var rad=a*Math.PI/180;
    var nr=r1*(0.85+0.3*Math.abs(Math.sin(a*7*Math.PI/180)));
    CIRC_IN.push({x:cx+nr*Math.cos(rad), y:cy+nr*Math.sin(rad)});
    var or=r2*(0.9+0.2*Math.abs(Math.cos(a*5*Math.PI/180)));
    CIRC_OUT.push({x:cx+or*Math.cos(rad), y:cy+or*Math.sin(rad)});
  });
})();

/* 3-class dataset for multi-class tab */
var CLS3=[
  {pts:[{x:2,y:7},{x:1.5,y:5.5},{x:3,y:8},{x:2.5,y:6},{x:1,y:7.5}],  col:"#38bdf8",label:"Class A"},
  {pts:[{x:5,y:9},{x:6,y:8},{x:5.5,y:7},{x:4.5,y:8.5},{x:7,y:9}],    col:"#fb923c",label:"Class B"},
  {pts:[{x:8,y:3},{x:7,y:4.5},{x:9,y:2},{x:8.5,y:5},{x:7.5,y:2.5}],  col:"#a78bfa",label:"Class C"}
];

/* ─── STATE ─── */
var S={
  tab:0,
  theta: Math.PI/2,   /* normal vector angle — π/2 → vertical boundary */
  bias: -5,
  C_val: 1.0,
  kernel: 0,          /* 0=linear  1=rbf */
  mcStrat: 0          /* 0=OvO     1=OvR */
};

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
    +'<div style="font-size:10px;color:'+C.muted+';width:72px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:56px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
}

/* ─── SVG SCAFFOLD ─── */
var VW=520,VH=340,PL=52,PR=18,PT=18,PB=44;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x){return PL+((x)/10)*PW;}
function sy(y){return PT+PH-((y)/10)*PH;}
function classify(x,y,w1,w2,b){return w1*x+w2*y+b;}

function plotAxes(xl,yl){
  var o='';
  [0,2,4,6,8,10].forEach(function(v){
    o+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,2,4,6,8,10].forEach(function(v){
    o+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';
    o+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';
  });
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||VW)+' '+(h||VH)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* clip a line (w1·x + w2·y + b = 0) to the plot region [0,10]×[0,10] */
function clipLine(w1,w2,b){
  var segs=[];
  if(Math.abs(w2)>1e-6){
    var y0=-b/w2;                       /* x=0 */
    if(y0>=0&&y0<=10) segs.push({x:0,y:y0});
    var y1=-(w1*10+b)/w2;               /* x=10 */
    if(y1>=0&&y1<=10) segs.push({x:10,y:y1});
  }
  if(Math.abs(w1)>1e-6){
    var x0=-b/w1;                       /* y=0 */
    if(x0>=0&&x0<=10&&segs.length<2) segs.push({x:x0,y:0});
    var x1=-(w2*10+b)/w1;              /* y=10 */
    if(x1>=0&&x1<=10&&segs.length<2) segs.push({x:x1,y:10});
  }
  var u=[];
  segs.forEach(function(s){
    if(!u.some(function(p){return Math.abs(p.x-s.x)<0.01&&Math.abs(p.y-s.y)<0.01;})) u.push(s);
  });
  if(u.length<2) return null;
  return {x1:sx(u[0].x),y1:sy(u[0].y),x2:sx(u[1].x),y2:sy(u[1].y)};
}
function drawLine(w1,w2,b,col,sw,dash,op){
  var L=clipLine(w1,w2,b); if(!L) return '';
  return '<line x1="'+L.x1.toFixed(1)+'" y1="'+L.y1.toFixed(1)+'"'
        +' x2="'+L.x2.toFixed(1)+'" y2="'+L.y2.toFixed(1)+'"'
        +' stroke="'+col+'" stroke-width="'+(sw||2)+'"'
        +(dash?' stroke-dasharray="'+dash+'"':'')+' opacity="'+(op||1)+'"/>';
}
function legend(items){
  return '<div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap;">'
    +items.map(function(i){
      return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
        +(i.line
          ?'<div style="width:18px;height:2px;background:'+i.col+';border-radius:1px;'+(i.dash?'background:repeating-linear-gradient(90deg,'+i.col+' 0,'+i.col+' 4px,transparent 4px,transparent 7px);':'')+'">'
          +'</div>'
          :'<div style="width:10px;height:10px;border-radius:'+(i.sq?'2px':'50%')+';background:'+i.col+';'+(i.ring?'border:2px dashed '+i.ring+';background:transparent;':'')+';"></div>')
        +i.label+'</div>';
    }).join('')
    +'</div>';
}

/* ═══════════════════════════════════════════════════
   TAB 0 — INTRO & DECISION BOUNDARY
═══════════════════════════════════════════════════ */
function renderIntro(){
  var th=S.theta, bi=S.bias;
  /* normalise weight vector */
  var w1=Math.cos(th), w2=Math.sin(th);
  var nrm=Math.sqrt(w1*w1+w2*w2); w1/=nrm; w2/=nrm;

  var cA=PTS_A.filter(function(p){return classify(p.x,p.y,w1,w2,bi)>=0;}).length;
  var cB=PTS_B.filter(function(p){return classify(p.x,p.y,w1,w2,bi)<0;}).length;
  var total=PTS_A.length+PTS_B.length;
  var acc=(cA+cB)/total*100;
  var accCol=acc>90?C.green:acc>70?C.yellow:C.red;

  var sv=plotAxes('Feature x\u2081','Feature x\u2082');

  /* half-plane shading */
  var pts2d=[{x:0,y:0},{x:10,y:0},{x:10,y:10},{x:0,y:10}];
  var polyA=pts2d.filter(function(p){return classify(p.x,p.y,w1,w2,bi)>=0;});
  if(polyA.length>0){
    var pA=polyA.map(function(p){return sx(p.x).toFixed(1)+','+sy(p.y).toFixed(1);}).join(' ');
    sv+='<polygon points="'+pA+'" fill="'+hex(C.blue,0.05)+'" stroke="none"/>';
  }

  /* decision boundary + label */
  sv+=drawLine(w1,w2,bi,C.accent,2.5,'','1');
  var L=clipLine(w1,w2,bi);
  if(L){
    var mx=(L.x1+L.x2)/2, my=(L.y1+L.y2)/2;
    sv+='<rect x="'+(mx-44)+'" y="'+(my-14)+'" width="88" height="15" rx="3" fill="#0a0a0f" opacity="0.85"/>';
    sv+='<text x="'+mx.toFixed(1)+'" y="'+(my-3).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="11" font-family="monospace" font-weight="700">decision boundary</text>';
  }

  /* data points */
  PTS_A.forEach(function(p){
    var ok=classify(p.x,p.y,w1,w2,bi)>=0;
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.blue+'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'" opacity="'+(ok?1:0.75)+'"/>';
    if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-9).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="12">&#10007;</text>';
  });
  PTS_B.forEach(function(p){
    var ok=classify(p.x,p.y,w1,w2,bi)<0;
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.orange+'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'" opacity="'+(ok?1:0.75)+'"/>';
    if(!ok) sv+='<text x="'+sx(p.x).toFixed(1)+'" y="'+(sy(p.y)-9).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="12">&#10007;</text>';
  });

  /* accuracy badge */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="124" height="21" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+16)+'" fill="'+accCol+'" font-size="12" font-family="monospace" font-weight="700">accuracy: '+acc.toFixed(0)+'%</text>';

  var out=sectionTitle('What is an SVM?','Find the hyperplane that separates two classes with the maximum possible margin');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* ── left: interactive plot ── */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Decision Boundary')
    +svgBox(sv)
    +legend([
      {col:C.blue,   label:'Class A (+1)'},
      {col:C.orange, label:'Class B (&#8722;1)'},
      {col:C.accent, label:'Boundary',line:true}
    ])
    +sliderRow('svmTheta',S.theta,0,Math.PI,0.02,'angle &#952;',2)
    +sliderRow('svmBias', S.bias,-9,-1,0.1,'bias b',1)
  );
  out+='</div>';

  /* ── right: stats + concept ── */
  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CLASSIFICATION STATS');
  out+=statRow('Total points',total,C.text);
  out+=statRow('Class A correct',cA+' / '+PTS_A.length,C.blue);
  out+=statRow('Class B correct',cB+' / '+PTS_B.length,C.orange);
  out+=statRow('Accuracy',acc.toFixed(1)+'%',accCol);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','THE CORE IDEA');
  out+=div('font-size:9px;color:'+C.text+';line-height:1.9;',
    'SVMs find the hyperplane that <span style="color:'+C.yellow+';font-weight:700;">maximises the gap</span> between the two classes — '
    +'not just any separating line.<br><br>'
    +'Decision function:<br>'
    +'<span style="color:'+C.purple+';font-family:monospace;font-size:10px;">f(x) = sign(w&#183;x + b)</span><br><br>'
    +'Only <span style="color:'+C.accent+';font-weight:700;">support vectors</span> (boundary points) influence the final hyperplane.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','EQUATION (2-D)');
  [
    {eq:'w\u2081x\u2081 + w\u2082x\u2082 + b = 0',lbl:'boundary (= 0)',col:C.accent},
    {eq:'w\u2081x\u2081 + w\u2082x\u2082 + b \u2265 +1',lbl:'class A slab',col:C.blue},
    {eq:'w\u2081x\u2081 + w\u2082x\u2082 + b \u2264 \u22121',lbl:'class B slab',col:C.orange},
  ].forEach(function(e){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;background:#08080d;border-left:3px solid '+e.col+';">'
      +'<div style="font-size:9px;font-family:monospace;color:'+e.col+';">'+e.eq+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+e.lbl+'</div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#129517;','One Line to Rule Them All',
    'Unlike Logistic Regression, SVM does <span style="color:'+C.red+';font-weight:700;">not</span> try to fit probabilities — '
    +'it purely seeks the <span style="color:'+C.accent+';font-weight:700;">widest street</span> between the two classes. '
    +'Adjust the sliders above: notice accuracy can be 100% with many different lines, '
    +'but only <span style="color:'+C.yellow+';font-weight:700;">one maximises the margin</span>.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════
   TAB 1 — MARGIN & SUPPORT VECTORS
═══════════════════════════════════════════════════ */
function renderMargin(){
  var norm=Math.sqrt(TRUE_W1*TRUE_W1+TRUE_W2*TRUE_W2);
  var margin=2/norm;

  var sv=plotAxes('Feature x\u2081','Feature x\u2082');

  /* half-plane shading */
  sv+='<rect x="'+PL+'" y="'+PT+'" width="'+(sx(5)-PL).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.blue,0.05)+'"/>';
  sv+='<rect x="'+sx(5).toFixed(1)+'" y="'+PT+'" width="'+(PL+PW-sx(5)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.orange,0.05)+'"/>';
  /* margin band */
  sv+='<rect x="'+sx(4).toFixed(1)+'" y="'+PT+'" width="'+(sx(6)-sx(4)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.yellow,0.06)+'"/>';

  /* slab boundaries */
  sv+=drawLine(1,0,-4,C.blue, 1.5,'6,3',0.7);
  sv+=drawLine(1,0,-6,C.orange,1.5,'6,3',0.7);
  /* decision boundary */
  sv+=drawLine(1,0,-5,C.accent,2.5,'',1);

  /* margin double-arrow at y=5 */
  var ay=sy(5);
  var defs='<defs>'
    +'<marker id="ah" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">'
    +'<path d="M0,0 L6,3 L0,6 Z" fill="'+C.yellow+'"/></marker>'
    +'<marker id="at" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse">'
    +'<path d="M0,0 L6,3 L0,6 Z" fill="'+C.yellow+'"/></marker>'
    +'</defs>';
  sv+='<line x1="'+sx(4).toFixed(1)+'" y1="'+ay.toFixed(1)+'" x2="'+sx(6).toFixed(1)+'" y2="'+ay.toFixed(1)+'"'
      +' stroke="'+C.yellow+'" stroke-width="1.5" marker-end="url(#ah)" marker-start="url(#at)"/>';
  sv+='<rect x="'+(sx(5)-32)+'" y="'+(ay-20)+'" width="64" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
  sv+='<text x="'+sx(5).toFixed(1)+'" y="'+(ay-9).toFixed(1)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="11" font-weight="700">&#947; = '+margin.toFixed(2)+'</text>';

  /* support vectors — dashed ring + filled dot + projection */
  SUPPORT_A.concat(SUPPORT_B).forEach(function(p){
    var col=p.x<5?C.blue:C.orange;
    sv+='<line x1="'+sx(p.x).toFixed(1)+'" y1="'+sy(p.y).toFixed(1)+'" x2="'+sx(5).toFixed(1)+'" y2="'+sy(p.y).toFixed(1)+'"'
        +' stroke="'+col+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.45"/>';
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="10" fill="none"'
        +' stroke="'+col+'" stroke-width="2" stroke-dasharray="4,2"/>';
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+col+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  });

  /* non-SV data points */
  PTS_A.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5.5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5" opacity="0.65"/>';
  });
  PTS_B.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5.5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5" opacity="0.65"/>';
  });

  /* slab equation labels */
  sv+='<text x="'+sx(2.3).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.blue+'" font-size="11" font-family="monospace" font-weight="700">w&#183;x+b = +1</text>';
  sv+='<text x="'+sx(7.7).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.orange+'" font-size="11" font-family="monospace" font-weight="700">w&#183;x+b = &#8722;1</text>';
  sv+='<text x="'+sx(5).toFixed(1)+'" y="'+(PT+15)+'" text-anchor="middle" fill="'+C.accent+'" font-size="11" font-family="monospace" font-weight="700">= 0</text>';

  var out=sectionTitle('Margin & Support Vectors','SVM maximises the margin — the gap between the two class boundaries');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Optimal Hyperplane')
    +svgBox(defs+sv)
    +legend([
      {col:C.blue,   label:'Class A (+1)'},
      {col:C.orange, label:'Class B (&#8722;1)'},
      {ring:C.blue,  col:'transparent',label:'Support vector A'},
      {ring:C.orange,col:'transparent',label:'Support vector B'},
    ])
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','GEOMETRY');
  out+=statRow('Decision boundary','x\u2081 = 5',C.accent);
  out+=statRow('Margin &#947;','2 / &#8214;w&#8214; = '+margin.toFixed(2),C.yellow);
  out+=statRow('Support vectors A',SUPPORT_A.length+' points',C.blue);
  out+=statRow('Support vectors B',SUPPORT_B.length+' points',C.orange);
  out+=statRow('&#8214;w&#8214;',norm.toFixed(3),C.purple);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','OPTIMISATION PROBLEM');
  [
    {eq:'Minimise:   \u00bd &#8214;w&#8214;\u00b2',        col:C.accent},
    {eq:'Subject to: y\u1d35(w\u00b7x\u1d35 + b) \u2265 1',col:C.yellow},
    {eq:'\u2200 i = 1, \u2026, n',                          col:C.muted},
  ].forEach(function(e){
    out+='<div style="font-size:9px;font-family:monospace;color:'+e.col+';padding:3px 0;">'+e.eq+'</div>';
  });
  out+='<div style="margin-top:10px;font-size:9px;color:'+C.muted+';line-height:1.7;">'
    +'Maximising margin &#8660; minimising &#8214;w&#8214;. Solved via <span style="color:'+C.purple+';">quadratic programming</span> or the dual (Lagrangian) formulation.'
    +'</div>';
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','DUAL FORM (LAGRANGIAN)');
  [
    {eq:'Maximise: &#931;\u03b1\u1d35 \u2212 \u00bd &#931;\u03b1\u1d35\u03b1\u2c7c y\u1d35y\u2c7c x\u1d35\u1d40x\u2c7c',col:C.accent},
    {eq:'s.t.  \u03b1\u1d35 \u2265 0,  &#931;\u03b1\u1d35y\u1d35 = 0',                                                     col:C.yellow},
    {eq:'w = &#931;\u03b1\u1d35 y\u1d35 x\u1d35  (only SVs have \u03b1>0)',                                                 col:C.blue},
  ].forEach(function(e){
    out+='<div style="font-size:8.5px;font-family:monospace;color:'+e.col+';padding:3px 0;">'+e.eq+'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','Why Only Support Vectors Matter',
    'Remove or move any <span style="color:'+C.green+';font-weight:700;">non-support-vector</span> point and the hyperplane doesn\'t change. '
    +'The dual formulation assigns <span style="color:'+C.yellow+';font-family:monospace;">\u03b1 = 0</span> to all other points. '
    +'This gives SVMs strong <span style="color:'+C.accent+';font-weight:700;">robustness to outliers</span> far from the margin boundary.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════
   TAB 2 — SOFT MARGIN (C PARAMETER)
═══════════════════════════════════════════════════ */
function renderSoftMargin(){
  var Cv=S.C_val;
  /* as C \u2192 \u221e: hard margin; as C \u2192 0: wide margin with violations */
  var halfM=Math.min(2.5, 1+2.5/Cv);
  var boundary=5.0;

  /* noisy boundary-crossing points */
  var noisePts=[
    {x:5.4,y:3.5,cls:1},{x:5.7,y:7.2,cls:1},{x:4.5,y:2.8,cls:1},
    {x:4.3,y:6.8,cls:-1},{x:4.7,y:2.2,cls:-1},{x:5.3,y:8.3,cls:-1}
  ];

  var sv=plotAxes('Feature x\u2081','Feature x\u2082');
  var xb=sx(boundary), xL=sx(boundary-halfM), xR=sx(boundary+halfM);

  /* shading */
  sv+='<rect x="'+PL+'" y="'+PT+'" width="'+(xb-PL).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.blue,0.04)+'"/>';
  sv+='<rect x="'+xb.toFixed(1)+'" y="'+PT+'" width="'+(PL+PW-xb).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.orange,0.04)+'"/>';
  sv+='<rect x="'+xL.toFixed(1)+'" y="'+PT+'" width="'+(xR-xL).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.yellow,0.06)+'"/>';

  /* slab + boundary lines */
  sv+=drawLine(1,0,-(boundary-halfM),C.blue,  1.5,'6,3',0.65);
  sv+=drawLine(1,0,-(boundary+halfM),C.orange,1.5,'6,3',0.65);
  sv+=drawLine(1,0,-boundary,        C.accent,2.5,'',   1);

  /* margin label */
  var my=sy(5);
  sv+='<line x1="'+xL.toFixed(1)+'" y1="'+my.toFixed(1)+'" x2="'+xR.toFixed(1)+'" y2="'+my.toFixed(1)+'" stroke="'+C.yellow+'" stroke-width="1.5"/>';
  sv+='<rect x="'+(xb-28)+'" y="'+(my-19)+'" width="56" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>';
  sv+='<text x="'+xb.toFixed(1)+'" y="'+(my-8).toFixed(1)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="11" font-weight="700">&#947; = '+(2*halfM).toFixed(2)+'</text>';

  /* main data points */
  PTS_A.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5.5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5" opacity="0.75"/>';
  });
  PTS_B.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="5.5" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5" opacity="0.75"/>';
  });

  /* noisy overlapping points — show slack when C is low */
  noisePts.forEach(function(p){
    var col=p.cls>0?C.blue:C.orange;
    var inViolation=(p.cls>0&&p.x>boundary-halfM)||(p.cls<0&&p.x<boundary+halfM);
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="4.5" fill="'+col
        +'" stroke="'+C.red+'" stroke-width="2" opacity="0.9"/>';
    /* slack arrow to nearest slab boundary */
    if(Cv<20){
      var target=p.cls>0?(boundary-halfM):(boundary+halfM);
      var penside=p.cls>0?(p.x>target):(p.x<target);
      if(penside){
        sv+='<line x1="'+sx(p.x).toFixed(1)+'" y1="'+sy(p.y).toFixed(1)+'"'
            +' x2="'+sx(target).toFixed(1)+'" y2="'+sy(p.y).toFixed(1)+'"'
            +' stroke="'+C.red+'" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.6"/>';
        var slk=Math.abs(p.x-target).toFixed(1);
        sv+='<text x="'+(sx(p.x)+sx(target))/2+'" y="'+(sy(p.y)-6).toFixed(1)+'"'
            +' text-anchor="middle" fill="'+C.red+'" font-size="12" font-family="monospace">\u03be='+slk+'</text>';
      }
    }
  });

  var out=sectionTitle('Soft Margin & C Parameter','C controls the tradeoff between a wide margin and classification errors');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Soft Margin Visualisation')
    +svgBox(sv)
    +legend([
      {col:C.blue,  label:'Class A'},
      {col:C.orange,label:'Class B'},
      {col:C.red,   label:'Violation (\u03be > 0)',line:true,dash:true}
    ])
    +sliderRow('svmC',S.C_val,0.01,50,0.01,'C value',2)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.green+';">\u2190 large margin, more slack</span>'
    +'<span style="color:'+C.red+';">strict, small margin \u2192</span>'
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT SETTINGS');
  var violLvl=Cv<1?'Many':Cv<10?'Some':'None';
  var violCol=Cv<1?C.red:Cv<10?C.yellow:C.green;
  var ofRisk=Cv>20?'High':Cv>5?'Medium':'Low';
  var ofCol=Cv>20?C.red:Cv>5?C.yellow:C.green;
  out+=statRow('C value',Cv.toFixed(2),C.accent);
  out+=statRow('Margin \u03b3',(2*halfM).toFixed(2),C.yellow);
  out+=statRow('Violations allowed',violLvl,violCol);
  out+=statRow('Overfitting risk',ofRisk,ofCol);
  out+=statRow('Bias',Cv<2?'Higher':'Lower',Cv<2?C.orange:C.green);
  out+=statRow('Variance',Cv>10?'Higher':'Lower',Cv>10?C.orange:C.green);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','SOFT MARGIN OBJECTIVE');
  [
    {eq:'Minimise:  \u00bd\u2016w\u2016\u00b2 + C \u00b7 \u2211\u03be\u1d35',  col:C.accent},
    {eq:'s.t.  y\u1d35(w\u00b7x\u1d35+b) \u2265 1 \u2212 \u03be\u1d35',        col:C.yellow},
    {eq:'      \u03be\u1d35 \u2265 0   (\u201cslack variable\u201d)',            col:C.purple},
  ].forEach(function(e){
    out+='<div style="font-size:9px;font-family:monospace;color:'+e.col+';padding:3px 0;">'+e.eq+'</div>';
  });
  out+='<div style="margin-top:8px;font-size:9px;color:'+C.muted+';line-height:1.7;">'
    +'\u03be\u1d35 measures how far point i has crossed into or through the margin slab.'
    +'</div>';
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','C \u2192 \u221e vs C \u2192 0');
  [
    {lbl:'C = \u221e (hard margin)',desc:'no errors allowed; may not converge on noisy data',c:C.red},
    {lbl:'C = 1 (balanced)',      desc:'sklearn default; good starting point',c:C.yellow},
    {lbl:'C \u2192 0 (max slack)',  desc:'large margin; many violations OK; high bias',c:C.green},
  ].forEach(function(r){
    out+='<div style="padding:5px 8px;margin:3px 0;border-radius:6px;border:1px solid '+hex(r.c,0.35)+';background:'+hex(r.c,0.05)+'">'
      +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.lbl+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9878;','Tuning C in Practice',
    'Use <span style="color:'+C.yellow+';font-weight:700;">cross-validation</span> (GridSearchCV or RandomizedSearchCV) to find the best C. '
    +'Start with a log-scale grid: <span style="color:'+C.accent+';font-family:monospace;">C \u2208 {0.001, 0.01, 0.1, 1, 10, 100}</span>. '
    +'Always <span style="color:'+C.green+';font-weight:700;">standardise features</span> first — C is sensitive to the feature scale, '
    +'and SVMs in general require normalised inputs.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════
   TAB 3 — KERNEL TRICK
═══════════════════════════════════════════════════ */
function renderKernel(){
  var ki=S.kernel;
  var sv=plotAxes('Feature x\u2081','Feature x\u2082');

  if(ki===0){
    /* linear kernel — best straight line still fails */
    sv+=drawLine(0,1,-5,C.accent,2,'5,4',0.85);
    var L=clipLine(0,1,-5);
    if(L){
      var mx=(L.x1+L.x2)/2;
      sv+='<rect x="'+(mx+6)+'" y="'+(L.y1+L.y2)/2-13+'" width="40" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>';
      sv+='<text x="'+(mx+26)+'" y="'+(L.y1+L.y2)/2-3+'" text-anchor="middle" fill="'+C.red+'" font-size="11" font-family="monospace">fails!</text>';
    }
    CIRC_IN.forEach(function(p){
      var ok=p.y<5;
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.blue
          +'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'" opacity="'+(ok?1:0.8)+'"/>';
    });
    CIRC_OUT.forEach(function(p){
      var ok=p.y>=5;
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.orange
          +'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2.5)+'" opacity="'+(ok?1:0.8)+'"/>';
    });
    /* misclassification count badge */
    var mc=CIRC_IN.filter(function(p){return p.y>=5;}).length
          +CIRC_OUT.filter(function(p){return p.y<5;}).length;
    sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="132" height="21" rx="4" fill="#0a0a0f" opacity="0.9"/>';
    sv+='<text x="'+(PL+10)+'" y="'+(PT+16)+'" fill="'+C.red+'" font-size="12" font-family="monospace" font-weight="700">'+mc+' misclassified &#10007;</text>';
  } else {
    /* RBF kernel — circular decision boundary */
    var cx=sx(5), cy2=sy(5);
    var rPx=sx(7.5)-sx(5);
    var rIn=sx(6.8)-sx(5), rOut=sx(8.2)-sx(5);
    /* fill inside */
    sv+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" r="'+rPx.toFixed(1)+'" fill="'+hex(C.blue,0.07)+'" stroke="none"/>';
    /* slab circles */
    sv+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" r="'+rIn.toFixed(1)+'" fill="none" stroke="'+C.blue+'" stroke-width="1" stroke-dasharray="5,3" opacity="0.55"/>';
    sv+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" r="'+rOut.toFixed(1)+'" fill="none" stroke="'+C.orange+'" stroke-width="1" stroke-dasharray="5,3" opacity="0.55"/>';
    /* decision boundary */
    sv+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy2.toFixed(1)+'" r="'+rPx.toFixed(1)+'" fill="none" stroke="'+C.accent+'" stroke-width="2.5"/>';

    CIRC_IN.forEach(function(p){
      var d=Math.sqrt((p.x-5)*(p.x-5)+(p.y-5)*(p.y-5));
      var ok=d<2.5;
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.blue
          +'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2)+'" opacity="'+(ok?1:0.8)+'"/>';
    });
    CIRC_OUT.forEach(function(p){
      var d=Math.sqrt((p.x-5)*(p.x-5)+(p.y-5)*(p.y-5));
      var ok=d>=2.5;
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+C.orange
          +'" stroke="'+(ok?'#0a0a0f':C.red)+'" stroke-width="'+(ok?1.5:2)+'" opacity="'+(ok?1:0.8)+'"/>';
    });
    sv+='<rect x="'+(cx-46)+'" y="'+(cy2-11)+'" width="92" height="14" rx="3" fill="#0a0a0f" opacity="0.88"/>';
    sv+='<text x="'+cx.toFixed(1)+'" y="'+cy2.toFixed(1)+'" text-anchor="middle" fill="'+C.green+'" font-size="11" font-family="monospace" font-weight="700">&#10003; RBF separates perfectly!</text>';
    sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="120" height="21" rx="4" fill="#0a0a0f" opacity="0.9"/>';
    sv+='<text x="'+(PL+10)+'" y="'+(PT+16)+'" fill="'+C.green+'" font-size="12" font-family="monospace" font-weight="700">0 misclassified &#10003;</text>';
  }

  var out=sectionTitle('The Kernel Trick','Transform data into higher dimensions where linear separation becomes possible');
  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  out+=btnSel(0,ki,C.red,  '&#128308; Linear Kernel — fails',  'kernelSel');
  out+=btnSel(1,ki,C.green,'&#128994; RBF Kernel — succeeds',  'kernelSel');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',
        ki===0?'<span style="color:'+C.red+';">No line can separate inner from outer</span>'
              :'<span style="color:'+C.green+';">RBF maps data to separable space</span>')
    +svgBox(sv)
    +legend([
      {col:C.blue,  label:'Inner class (+1)'},
      {col:C.orange,label:'Outer class (&#8722;1)'},
      {col:C.accent,label:'Decision boundary',line:true}
    ])
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','COMMON KERNELS');
  [
    {name:'Linear',   eq:'K(x,z) = x\u1d40z',               use:'Linearly separable data',   c:ki===0?C.red:C.muted},
    {name:'Polynomial',eq:'K(x,z) = (x\u1d40z + c)\u1d48',  use:'Polynomial boundaries',     c:C.purple},
    {name:'RBF / Gaussian',eq:'K(x,z) = exp(\u2212\u03b3\u2016x\u2212z\u2016\u00b2)',use:'Any smooth boundary',c:ki===1?C.green:C.muted},
    {name:'Sigmoid',  eq:'K(x,z) = tanh(\u03b1x\u1d40z+c)', use:'Neural-net-like',           c:C.yellow},
  ].forEach(function(k){
    var active=k.c!==C.muted;
    out+='<div style="padding:6px 9px;margin:3px 0;border-radius:6px;border:1px solid '+(active?hex(k.c,0.45):C.border)+';background:'+(active?hex(k.c,0.06):'transparent')+'">'
      +'<div style="font-size:10px;font-weight:700;color:'+k.c+';">'+k.name+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';font-family:monospace;margin:1px 0;">'+k.eq+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';">'+k.use+'</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','RBF HYPERPARAMETER \u03b3');
  [
    {lbl:'High \u03b3',val:'narrow radius',desc:'each point has small influence; can overfit',c:C.red},
    {lbl:'Low \u03b3', val:'wide radius',  desc:'smoother, broader boundary; may underfit',c:C.blue},
    {lbl:'Default',    val:'"scale" mode', desc:'1/(n_features \u00b7 Var(X)) — good start',c:C.green},
  ].forEach(function(r){
    out+='<div style="padding:4px 7px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="display:flex;justify-content:space-between;font-size:9px;">'
      +'<span style="color:'+r.c+';font-weight:700;">'+r.lbl+'</span>'
      +'<span style="color:'+C.muted+';">'+r.val+'</span></div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">'+r.desc+'</div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127744;','The Trick: Never Leave 2-D',
    'Instead of explicitly mapping \u03c6(x), kernels compute '
    +'<span style="color:'+C.accent+';font-family:monospace;">K(x\u1d35, x\u2c7c) = \u03c6(x\u1d35)\u1d40\u03c6(x\u2c7c)</span> directly. '
    +'The RBF kernel corresponds to an <span style="color:'+C.yellow+';font-weight:700;">infinite-dimensional feature space</span> — '
    +'computed cheaply in O(n\u00b2) dot products. '
    +'This is why SVMs can fit <span style="color:'+C.green+';font-weight:700;">arbitrarily complex boundaries</span> without exploding memory.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════
   TAB 4 — MULTI-CLASS SVM
═══════════════════════════════════════════════════ */
function renderMulticlass(){
  var strat=S.mcStrat;
  var sv=plotAxes('Feature x\u2081','Feature x\u2082');

  if(strat===0){
    /* OvO — three pairwise boundaries */
    sv+=drawLine(0, 1,-8.0, C.accent, 2,'',0.85);          /* A vs B: y = 8 */
    sv+=drawLine(1, 1,-9.5, C.yellow, 2,'',0.85);          /* A vs C: x+y = 9.5 */
    sv+=drawLine(1,-1,-2,   C.purple, 2,'',0.85);          /* B vs C: x-y = 2 */
    /* boundary labels */
    var labAB=clipLine(0,1,-8.0);
    if(labAB) sv+='<rect x="'+(labAB.x1+4)+'" y="'+(labAB.y1-15)+'" width="32" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
                  +'<text x="'+(labAB.x1+20)+'" y="'+(labAB.y1-5)+'" text-anchor="middle" fill="'+C.accent+'" font-size="12" font-family="monospace">A|B</text>';
    var labAC=clipLine(1,1,-9.5);
    if(labAC){var mx2=(labAC.x1+labAC.x2)/2,my2=(labAC.y1+labAC.y2)/2;
      sv+='<rect x="'+(mx2+4)+'" y="'+(my2+2)+'" width="32" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
         +'<text x="'+(mx2+20)+'" y="'+(my2+12)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="12" font-family="monospace">A|C</text>';}
    var labBC=clipLine(1,-1,-2);
    if(labBC){var mx3=(labBC.x1+labBC.x2)/2,my3=(labBC.y1+labBC.y2)/2;
      sv+='<rect x="'+(mx3+4)+'" y="'+(my3+2)+'" width="32" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
         +'<text x="'+(mx3+20)+'" y="'+(my3+12)+'" text-anchor="middle" fill="'+C.purple+'" font-size="12" font-family="monospace">B|C</text>';}
  } else {
    /* OvR — three one-vs-rest boundaries */
    sv+=drawLine( 1, 1,-9.5,  C.blue,  2,'',0.85);  /* A vs rest */
    sv+=drawLine(-1, 2,-8,    C.orange,2,'',0.85);  /* B vs rest */
    sv+=drawLine( 2, 1,-18,   C.purple,2,'',0.85);  /* C vs rest */
    var l1=clipLine(1,1,-9.5);
    if(l1) sv+='<rect x="'+(l1.x1-38)+'" y="'+(l1.y1-14)+'" width="36" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
               +'<text x="'+(l1.x1-20)+'" y="'+(l1.y1-4)+'" text-anchor="middle" fill="'+C.blue+'" font-size="12" font-family="monospace">A|rest</text>';
    var l2=clipLine(-1,2,-8);
    if(l2){var lx2=(l2.x1+l2.x2)/2,ly2=(l2.y1+l2.y2)/2;
      sv+='<rect x="'+(lx2-38)+'" y="'+(ly2-14)+'" width="42" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
         +'<text x="'+(lx2-17)+'" y="'+(ly2-4)+'" text-anchor="middle" fill="'+C.orange+'" font-size="12" font-family="monospace">B|rest</text>';}
    var l3=clipLine(2,1,-18);
    if(l3){var lx3=(l3.x1+l3.x2)/2,ly3=(l3.y1+l3.y2)/2;
      sv+='<rect x="'+(lx3-40)+'" y="'+(ly3+2)+'" width="42" height="13" rx="3" fill="#0a0a0f" opacity="0.85"/>'
         +'<text x="'+(lx3-19)+'" y="'+(ly3+12)+'" text-anchor="middle" fill="'+C.purple+'" font-size="12" font-family="monospace">C|rest</text>';}
  }

  /* data points */
  CLS3.forEach(function(cls){
    cls.pts.forEach(function(p){
      sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="6.5" fill="'+cls.col+'" stroke="#0a0a0f" stroke-width="1.5"/>';
    });
  });

  var k=CLS3.length;
  var nOvO=k*(k-1)/2, nOvR=k;

  var out=sectionTitle('Multi-class SVM','Two strategies: One-vs-One (OvO) and One-vs-Rest (OvR)');
  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  out+=btnSel(0,strat,C.accent, '&#9876; One-vs-One (OvO)', 'mcStrat');
  out+=btnSel(1,strat,C.purple, '&#127382; One-vs-Rest (OvR)','mcStrat');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',
        strat===0?'<span style="color:'+C.accent+';">OvO: one classifier per class pair</span>'
                 :'<span style="color:'+C.purple+';">OvR: each class vs all others</span>')
    +svgBox(sv)
    +legend(CLS3.map(function(c){return {col:c.col,label:c.label};}))
    +'<div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">'
    +(strat===0
      ?[{c:C.accent,l:'A|B'},{c:C.yellow,l:'A|C'},{c:C.purple,l:'B|C'}]
      :[{c:C.blue,l:'A|rest'},{c:C.orange,l:'B|rest'},{c:C.purple,l:'C|rest'}]
    ).map(function(b){
      return '<div style="display:flex;align-items:center;gap:5px;font-size:8.5px;color:'+C.muted+';">'
        +'<div style="width:14px;height:2px;background:'+b.c+';border-radius:1px;"></div>'+b.l+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','COMPARISON (k=3 classes)');
  var rows=[
    {lbl:'Strategy',        ovo:'OvO',          ovr:'OvR',          cO:C.accent,cR:C.purple},
    {lbl:'# classifiers',   ovo:nOvO+' = k(k-1)/2',ovr:nOvR+' = k',  cO:C.yellow,cR:C.yellow},
    {lbl:'Training set size',ovo:'2-class subset',ovr:'full dataset',  cO:C.text,  cR:C.text},
    {lbl:'Prediction rule',  ovo:'majority vote', ovr:'max score',    cO:C.blue,  cR:C.blue},
    {lbl:'sklearn default',  ovo:'&#10003; SVC', ovr:'LinearSVC',   cO:C.green, cR:C.green},
  ];
  var hdr='<div style="display:flex;font-size:9px;font-weight:700;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<div style="flex:1.4;color:'+C.muted+';">Property</div>'
    +'<div style="flex:1;color:'+C.accent+';text-align:center;">OvO</div>'
    +'<div style="flex:1;color:'+C.purple+';text-align:center;">OvR</div>'
    +'</div>';
  out+=hdr;
  rows.forEach(function(r){
    out+='<div style="display:flex;font-size:9px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:1.4;color:'+C.muted+';">'+r.lbl+'</div>'
      +'<div style="flex:1;color:'+r.cO+';font-weight:700;text-align:center;">'+r.ovo+'</div>'
      +'<div style="flex:1;color:'+r.cR+';font-weight:700;text-align:center;">'+r.ovr+'</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','SCALING WITH k CLASSES');
  [[3,3,3],[5,10,5],[10,45,10],[50,1225,50]].forEach(function(r){
    out+='<div style="display:flex;justify-content:space-between;font-size:9px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+C.muted+';">k = '+r[0]+'</span>'
      +'<span style="color:'+C.accent+';font-family:monospace;">OvO: '+r[1]+'</span>'
      +'<span style="color:'+C.purple+';font-family:monospace;">OvR: '+r[2]+'</span>'
      +'</div>';
  });
  out+='<div style="font-size:8.5px;color:'+C.dim+';margin-top:8px;line-height:1.6;">OvO grows quadratically (k\u00b2/2); OvR grows linearly (k). For large k, OvR is preferred.</div>';
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128347;','OvO vs OvR in Practice',
    '<span style="color:'+C.accent+';font-weight:700;">OvO</span> trains on smaller datasets (2-class at a time) — often faster per classifier and preferred by '
    +'<span style="color:'+C.yellow+';font-family:monospace;">sklearn.SVC</span>. '
    +'<span style="color:'+C.purple+';font-weight:700;">OvR</span> trains fewer classifiers but on the full imbalanced dataset — preferred by '
    +'<span style="color:'+C.yellow+';font-family:monospace;">LinearSVC</span>. '
    +'With <span style="color:'+C.green+';font-weight:700;">well-calibrated scores</span> both strategies give similar accuracy; choose based on class balance and k size.'
  );
  return out;
}

/* ═══════════════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════════════ */
var TABS=[
  '&#128208; Intro &amp; Boundary',
  '&#128207; Margin &amp; SVs',
  '&#9878; Soft Margin (C)',
  '&#127744; Kernel Trick',
  '&#128347; Multi-class'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.purple+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">'
    +'Support Vector Machine</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough &#8212; from hyperplanes and margins to kernels and multi-class strategies')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderIntro();
  else if(S.tab===1) html+=renderMargin();
  else if(S.tab===2) html+=renderSoftMargin();
  else if(S.tab===3) html+=renderKernel();
  else if(S.tab===4) html+=renderMulticlass();
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
        if(action==='tab')       {S.tab=idx;     render();}
        else if(action==='kernelSel'){S.kernel=idx; render();}
        else if(action==='mcStrat') {S.mcStrat=idx;render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='svmTheta'){S.theta=val; render();}
        else if(action==='svmBias'){S.bias=val;  render();}
        else if(action==='svmC')  {S.C_val=val; render();}
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

SVM_VISUAL_HEIGHT = 1300
