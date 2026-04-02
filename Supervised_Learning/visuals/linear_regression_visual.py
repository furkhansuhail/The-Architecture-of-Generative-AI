"""
Self-contained HTML visual for Linear Regression.
5 interactive tabs: Intro & Fitting, Cost Function, Gradient Descent,
Evaluation (R²), Multiple Regression.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(LR_VISUAL_HTML, height=LR_VISUAL_HEIGHT, scrolling=True)
"""

LR_VISUAL_HTML = r"""<!DOCTYPE html>
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
var PTS=[
  {x:1.2,y:2.1},{x:2.0,y:3.8},{x:2.8,y:4.2},{x:3.5,y:5.9},
  {x:4.1,y:6.1},{x:4.9,y:7.5},{x:5.6,y:8.0},{x:6.2,y:9.3},
  {x:7.0,y:10.1},{x:7.8,y:11.4},{x:8.5,y:12.0},{x:9.1,y:13.2}
];
var TM=1.42,TB=0.6; /* true slope & intercept */

/* ─── MATH ─── */
function mseAt(m,b){
  var s=0;
  PTS.forEach(function(p){var e=p.y-(m*p.x+b);s+=e*e;});
  return s/PTS.length;
}
function gradM(m,b){
  var s=0;
  PTS.forEach(function(p){var e=p.y-(m*p.x+b);s+=-2*p.x*e;});
  return s/PTS.length;
}

/* ─── STATE ─── */
var S={
  tab:0,
  slope:0.8, intercept:1.0,
  costSlope:3.2,
  gdHist:[{m:3.2,mse:mseAt(3.2,TB)}],
  gdLr:0.3,
  gdRunning:false,
  gdTimer:null,
  noise:1.2,
  mlSel:0
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
    +'<div style="font-size:10px;color:'+C.accent+';width:48px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
var PW=VW-PL-PR,PH=VH-PT-PB;
function sx(x){return PL+((x)/11)*PW;}
function sy(y){return PT+PH-((y)/16)*PH;}

function plotAxes(xlabel,ylabel){
  var o='';
  [0,2,4,6,8,10].forEach(function(v){
    o+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  [0,4,8,12,16].forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,2,4,6,8,10].forEach(function(v){
    o+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  [0,4,8,12,16].forEach(function(v){
    o+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xlabel+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+ylabel+'</text>';
  return o;
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||VW)+' '+(h||VH)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ═══════════════════════════════════════════
   TAB 1 — INTRO & FITTING
═══════════════════════════════════════════ */
function renderIntro(){
  var m=S.slope,b=S.intercept;
  var mse=mseAt(m,b);
  var best=mseAt(TM,TB);
  var mseCol=mse<best*1.2?C.green:mse<best*3?C.yellow:C.red;
  var barW=Math.max(4,Math.min(100,(1-(mse-best)/25)*100));

  var sv=plotAxes('Feature (x)','Target (y)');
  /* residuals */
  PTS.forEach(function(p){
    var yh=m*p.x+b;
    sv+='<line x1="'+sx(p.x).toFixed(1)+'" y1="'+sy(p.y).toFixed(1)+'" x2="'+sx(p.x).toFixed(1)+'" y2="'+sy(yh).toFixed(1)+'" stroke="'+C.red+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>';
  });
  /* true best-fit */
  sv+='<line x1="'+sx(0)+'" y1="'+sy(TB).toFixed(1)+'" x2="'+sx(10)+'" y2="'+sy(TM*10+TB).toFixed(1)+'" stroke="'+C.green+'" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.6"/>';
  sv+='<text x="'+sx(8.8).toFixed(1)+'" y="'+(sy(TM*8.8+TB)-8).toFixed(1)+'" fill="'+C.green+'" font-size="8" text-anchor="middle" font-family="monospace">best fit</text>';
  /* user line */
  sv+='<line x1="'+sx(0)+'" y1="'+sy(b).toFixed(1)+'" x2="'+sx(10)+'" y2="'+sy(m*10+b).toFixed(1)+'" stroke="'+C.accent+'" stroke-width="2.5"/>';
  /* points */
  PTS.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="4" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  });
  /* equation box */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="155" height="21" rx="4" fill="#0a0a0f" opacity="0.88"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+16)+'" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">y&#770; = '+m.toFixed(2)+'x + ('+b.toFixed(2)+')</text>';

  var out=sectionTitle('What is Linear Regression?','Find the line that best explains the relationship between x and y');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* ── left: plot ── */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Fit')
    +svgBox(sv)
    +sliderRow('slope',m,-0.5,3,0.05,'slope m',2)
    +sliderRow('intercept',b,-4,6,0.1,'intercept b',1)
  );
  out+='</div>';

  /* ── right: info cards ── */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';

  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','THE EQUATION')
    +div('font-size:22px;font-weight:800;color:'+C.accent+';font-family:monospace;text-align:center;margin:8px 0;','y = mx + b')
    +div('font-size:9px;color:'+C.muted+';line-height:1.9;',
      '<div><span style="color:'+C.accent+';font-weight:700;">y</span> &#8212; predicted output</div>'
      +'<div><span style="color:'+C.blue+';font-weight:700;">x</span> &#8212; input feature</div>'
      +'<div><span style="color:'+C.yellow+';font-weight:700;">m</span> &#8212; slope (weight)</div>'
      +'<div><span style="color:'+C.purple+';font-weight:700;">b</span> &#8212; intercept (bias)</div>'
    ),'max-width:none;margin:0 0 12px;'
  );

  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','YOUR LINE vs BEST FIT')
    +'<div style="text-align:center;margin:8px 0;">'
    +div('font-size:11px;color:'+C.muted+';margin-bottom:4px;','Your MSE')
    +div('font-size:28px;font-weight:800;color:'+mseCol+';transition:color .3s;',mse.toFixed(2))
    +div('font-size:9px;color:'+C.dim+';margin-top:4px;','Best possible: '+best.toFixed(2))
    +'</div>'
    +'<div style="height:8px;border-radius:4px;background:'+C.border+';overflow:hidden;">'
    +'<div style="height:100%;width:'+barW+'%;border-radius:4px;background:'+mseCol+';transition:all .3s;"></div></div>'
    ,'max-width:none;margin:0 0 12px;'
  );

  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','RESIDUALS')
    +div('font-size:10px;color:'+C.muted+';line-height:1.8;',
      'The <span style="color:'+C.red+'">red dashed lines</span> are <b style="color:'+C.text+'">residuals</b>: the vertical gap between each data point and the predicted line.'
      +'<div style="margin-top:8px;font-family:monospace;">e&#7522; = y&#7522; &#8722; y&#770;&#7522;</div>'
    ),'max-width:none;margin:0;'
  );

  out+='</div></div>';
  out+=insight('&#128161;','The Core Idea',
    'Linear regression finds the <span style="color:'+C.yellow+';font-weight:700;">m</span> and <span style="color:'+C.purple+';font-weight:700;">b</span> that minimize total squared residuals. '
    +'Drag the sliders &#8212; the <span style="color:'+C.green+'">green dashed line</span> shows the mathematically optimal fit.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 2 — COST FUNCTION
═══════════════════════════════════════════ */
function renderCost(){
  var m=S.costSlope;
  var curMse=mseAt(m,TB);
  var bestMse=mseAt(TM,TB);
  var mseCol=curMse<0.5?C.green:curMse<6?C.yellow:C.red;
  var mseStatus=curMse<0.5?'&#10003; near optimal!':curMse<6?'getting closer...':'&#9888; far from optimal';

  /* cost-curve SVG */
  var CW=440,CH=250,CL=52,CR=20,CTp=20,CBt=40;
  var CPW=CW-CL-CR,CPH=CH-CTp-CBt;
  var mMin=-0.5,mMax=3.5,cMax=80;
  function cx(v){return CL+((v-mMin)/(mMax-mMin))*CPW;}
  function cy(v){return CTp+CPH-Math.min(1,v/cMax)*CPH;}

  var sv='';
  /* grid */
  [-0.5,0.5,1.42,2.5,3.5].forEach(function(v){
    sv+='<line x1="'+cx(v).toFixed(1)+'" y1="'+CTp+'" x2="'+cx(v).toFixed(1)+'" y2="'+(CTp+CPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  [0,20,40,60,80].forEach(function(v){
    sv+='<line x1="'+CL+'" y1="'+cy(v).toFixed(1)+'" x2="'+(CL+CPW)+'" y2="'+cy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  sv+='<line x1="'+CL+'" y1="'+CTp+'" x2="'+CL+'" y2="'+(CTp+CPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+CL+'" y1="'+(CTp+CPH)+'" x2="'+(CL+CPW)+'" y2="'+(CTp+CPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  /* tick labels */
  [-0.5,0.5,1.42,2.5,3.5].forEach(function(v){
    var lbl=v===1.42?'m*':v.toFixed(1);
    var col=v===1.42?C.green:C.muted;
    var fw=v===1.42?'700':'400';
    sv+='<text x="'+cx(v).toFixed(1)+'" y="'+(CTp+CPH+13)+'" text-anchor="middle" fill="'+col+'" font-size="8" font-family="monospace" font-weight="'+fw+'">'+lbl+'</text>';
  });
  [0,20,40,60,80].forEach(function(v){
    sv+='<text x="'+(CL-6)+'" y="'+(cy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  sv+='<text x="'+(CL+CPW/2)+'" y="'+(CH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">slope (m)</text>';
  sv+='<text x="12" y="'+(CTp+CPH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,12,'+(CTp+CPH/2)+')">MSE (cost)</text>';
  /* parabola */
  var pts=[];
  for(var v=mMin;v<=mMax+0.01;v+=0.06){pts.push(cx(v).toFixed(1)+','+cy(mseAt(v,TB)).toFixed(1));}
  sv+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+C.purple+'" stroke-width="2.5"/>';
  /* minimum */
  sv+='<circle cx="'+cx(TM).toFixed(1)+'" cy="'+cy(bestMse).toFixed(1)+'" r="6" fill="'+C.green+'" opacity="0.9"/>';
  sv+='<text x="'+(cx(TM)+10).toFixed(1)+'" y="'+(cy(bestMse)-8).toFixed(1)+'" fill="'+C.green+'" font-size="9" font-family="monospace" font-weight="700">global minimum</text>';
  /* current */
  sv+='<line x1="'+cx(m).toFixed(1)+'" y1="'+CTp+'" x2="'+cx(m).toFixed(1)+'" y2="'+(CTp+CPH)+'" stroke="'+C.accent+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>';
  sv+='<circle cx="'+cx(m).toFixed(1)+'" cy="'+cy(curMse).toFixed(1)+'" r="7" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<text x="'+(CL+8)+'" y="'+(CTp+13)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">MSE = '+curMse.toFixed(2)+'</text>';

  /* bar chart: individual squared errors */
  var bsv='';
  PTS.forEach(function(p,i){
    var yh=m*p.x+TB;
    var sq=Math.pow(p.y-yh,2);
    var bh=Math.min(58,(sq/50)*58);
    var bx=16+i*54;
    var bc=sq<0.5?C.green:sq<5?C.yellow:C.red;
    bsv+='<rect x="'+bx+'" y="'+(68-bh).toFixed(1)+'" width="40" height="'+bh.toFixed(1)+'" rx="3" fill="'+bc+'" opacity="0.75"/>';
    bsv+='<text x="'+(bx+20)+'" y="78" text-anchor="middle" fill="'+C.muted+'" font-size="7" font-family="monospace">'+p.x.toFixed(1)+'</text>';
    if(sq>0.8) bsv+='<text x="'+(bx+20)+'" y="'+(68-bh-3).toFixed(1)+'" text-anchor="middle" fill="'+bc+'" font-size="7" font-family="monospace">'+sq.toFixed(1)+'</text>';
  });

  var out=sectionTitle('The Cost Function (MSE)','Measure how wrong the line is &#8212; then minimize it');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','MSE vs. Slope m')
    +'<svg width="100%" viewBox="0 0 '+CW+' '+CH+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv+'</svg>'
    +sliderRow('costSlope',m,-0.5,3.5,0.05,'slope m',2)
  );
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Individual Squared Errors')
    +'<svg width="100%" viewBox="0 0 680 84" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+bsv+'</svg>'
    +div('font-size:8px;color:'+C.muted+';margin-top:6px;text-align:center;','x values (each bar = (y&#7522; &#8722; y&#770;&#7522;)&#178;)')
  );
  out+='</div>';

  /* right */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','FORMULA')
    +'<div style="background:#08080d;border-radius:8px;padding:12px;text-align:center;margin-bottom:10px;">'
    +div('font-size:11px;color:'+C.purple+';font-weight:700;line-height:2;font-family:monospace;','MSE = (1/n)&#8721;(y&#7522;&#8722;y&#770;&#7522;)&#178;')
    +'</div>'
    +div('font-size:9px;color:'+C.muted+';line-height:1.9;',
      '<div><span style="color:'+C.text+';font-weight:700;">n</span> &#8212; number of data points</div>'
      +'<div><span style="color:'+C.blue+';font-weight:700;">y&#7522;</span> &#8212; actual value</div>'
      +'<div><span style="color:'+C.accent+';font-weight:700;">y&#770;&#7522;</span> &#8212; predicted value</div>'
      +'<div><span style="color:'+C.yellow+';font-weight:700;">squaring</span> &#8212; penalizes big errors</div>'
    ),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','CURRENT MSE')
    +'<div style="text-align:center;">'
    +div('font-size:36px;font-weight:800;color:'+mseCol+';transition:color .3s;font-family:monospace;',curMse.toFixed(2))
    +div('font-size:9px;color:'+C.dim+';margin-top:4px;',mseStatus)
    +'</div>'
    +'<div style="height:8px;border-radius:4px;background:'+C.border+';overflow:hidden;margin-top:8px;">'
    +'<div style="height:100%;width:'+Math.max(4,100-Math.min(100,(curMse/80)*100))+'%;border-radius:4px;background:'+mseCol+';transition:all .3s;"></div></div>'
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','WHY SQUARED?')
    +div('font-size:10px;color:'+C.muted+';line-height:1.8;',
      'Large mistakes are penalized <b style="color:'+C.text+'">much more heavily</b>. '
      +'Squaring also makes the cost <span style="color:'+C.purple+'">smooth and differentiable</span> &#8212; essential for calculus-based optimization.'
    ),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('&#127919;','The Parabola',
    'The cost forms a <span style="color:'+C.purple+';font-weight:700;">U-shaped convex parabola</span> &#8212; exactly one <span style="color:'+C.green+';font-weight:700;">global minimum</span>, no local traps. '
    +'Drag the slider all the way to <span style="color:'+C.accent+'">m*</span> to confirm!'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 3 — GRADIENT DESCENT
═══════════════════════════════════════════ */
function renderGradient(){
  var hist=S.gdHist;
  var cur=hist[hist.length-1];
  var lr=S.gdLr;
  var converged=hist.length>5&&Math.abs(cur.mse-mseAt(TM,TB))<0.06;

  var CW=440,CH=250,CL=52,CR=20,CTp=20,CBt=40;
  var CPW=CW-CL-CR,CPH=CH-CTp-CBt;
  var mMin=-0.5,mMax=3.8,cMax=80;
  function cx(v){return CL+((v-mMin)/(mMax-mMin))*CPW;}
  function cy(v){return CTp+CPH-Math.min(1,v/cMax)*CPH;}

  var sv='';
  [-0.5,0.5,1.42,2.5,3.5].forEach(function(v){
    sv+='<line x1="'+cx(v).toFixed(1)+'" y1="'+CTp+'" x2="'+cx(v).toFixed(1)+'" y2="'+(CTp+CPH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  [0,20,40,60,80].forEach(function(v){
    sv+='<line x1="'+CL+'" y1="'+cy(v).toFixed(1)+'" x2="'+(CL+CPW)+'" y2="'+cy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  sv+='<line x1="'+CL+'" y1="'+CTp+'" x2="'+CL+'" y2="'+(CTp+CPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+CL+'" y1="'+(CTp+CPH)+'" x2="'+(CL+CPW)+'" y2="'+(CTp+CPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [-0.5,0.5,1.42,2.5,3.5].forEach(function(v){
    var lbl=v===1.42?'m*':v.toFixed(1);
    sv+='<text x="'+cx(v).toFixed(1)+'" y="'+(CTp+CPH+13)+'" text-anchor="middle" fill="'+(v===1.42?C.green:C.muted)+'" font-size="8" font-family="monospace" font-weight="'+(v===1.42?'700':'400')+'">'+lbl+'</text>';
  });
  [0,20,40,60,80].forEach(function(v){
    sv+='<text x="'+(CL-6)+'" y="'+(cy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  sv+='<text x="'+(CL+CPW/2)+'" y="'+(CH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">slope (m)</text>';
  sv+='<text x="12" y="'+(CTp+CPH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,12,'+(CTp+CPH/2)+')">MSE</text>';
  /* parabola */
  var pts=[];
  for(var v=mMin;v<=mMax+0.01;v+=0.06){pts.push(cx(v).toFixed(1)+','+cy(mseAt(v,TB)).toFixed(1));}
  sv+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+C.purple+'" stroke-width="2.5" opacity="0.7"/>';
  /* minimum */
  sv+='<circle cx="'+cx(TM).toFixed(1)+'" cy="'+cy(mseAt(TM,TB)).toFixed(1)+'" r="5" fill="'+C.green+'" opacity="0.9"/>';
  /* descent path */
  for(var i=1;i<hist.length;i++){
    var p0=hist[i-1],p1=hist[i];
    sv+='<line x1="'+cx(p0.m).toFixed(1)+'" y1="'+cy(p0.mse).toFixed(1)+'" x2="'+cx(p1.m).toFixed(1)+'" y2="'+cy(p1.mse).toFixed(1)+'" stroke="'+C.accent+'" stroke-width="1.5" opacity="0.65"/>';
  }
  /* current dot */
  sv+='<circle cx="'+cx(cur.m).toFixed(1)+'" cy="'+cy(cur.mse).toFixed(1)+'" r="7" fill="'+(converged?C.green:C.accent)+'" stroke="#0a0a0f" stroke-width="2"/>';
  /* gradient arrow */
  if(cur.grad&&!converged){
    var dir=cur.grad>0?-1:1;
    var ax1=cx(cur.m),ay1=cy(cur.mse),ax2=ax1+dir*26,ay2=ay1;
    sv+='<defs><marker id="arh" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0,6 2,0 4" fill="'+C.orange+'"/></marker></defs>';
    sv+='<line x1="'+ax1.toFixed(1)+'" y1="'+ay1.toFixed(1)+'" x2="'+ax2.toFixed(1)+'" y2="'+ay2.toFixed(1)+'" stroke="'+C.orange+'" stroke-width="2" marker-end="url(#arh)"/>';
  }
  var stepLbl=converged?'&#10003; converged! m='+cur.m.toFixed(3):'step '+hist.length+' | m='+cur.m.toFixed(3);
  sv+='<text x="'+(CL+8)+'" y="'+(CTp+13)+'" fill="'+(converged?C.green:C.muted)+'" font-size="8" font-family="monospace" font-weight="700">'+stepLbl+'</text>';

  var curGradStr=cur.grad?cur.grad.toFixed(4):'&#8212;';

  var out=sectionTitle('Gradient Descent','Iteratively move m in the direction that reduces MSE fastest');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Descending the Cost Curve')
    +'<svg width="100%" viewBox="0 0 '+CW+' '+CH+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv+'</svg>'
    +'<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">'
    +'<button data-action="gdRun" style="padding:8px 16px;border-radius:8px;border:1px solid '+C.accent+';background:'+(S.gdRunning?hex(C.accent,.15):hex(C.accent,.08))+';color:'+C.accent+';font-size:10px;font-weight:700;font-family:inherit;">'
    +(S.gdRunning?'&#9646;&#9646; Pause':'&#9654; Run')+'</button>'
    +'<button data-action="gdStep" style="padding:8px 16px;border-radius:8px;border:1px solid '+C.blue+';background:'+hex(C.blue,.08)+';color:'+C.blue+';font-size:10px;font-weight:700;font-family:inherit;">+1 Step</button>'
    +'<button data-action="gdReset" style="padding:8px 16px;border-radius:8px;border:1px solid '+C.dim+';background:transparent;color:'+C.muted+';font-size:10px;font-weight:700;font-family:inherit;">&#8635; Reset</button>'
    +'</div>'
    +sliderRow('gdLr',lr,0.01,0.8,0.01,'&#945; lr',2)
  );
  out+='</div>';

  /* right */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','THE UPDATE RULE')
    +'<div style="background:#08080d;border-radius:8px;padding:12px;text-align:center;font-family:monospace;">'
    +div('font-size:11px;color:'+C.accent+';font-weight:700;line-height:2.2;','m := m &#8722; &#945; &#8706;J/&#8706;m')
    +div('font-size:8px;color:'+C.muted+';margin-top:4px;','&#8706;J/&#8706;m = &#8722;(2/n)&#8721;x&#7522;(y&#7522;&#8722;y&#770;&#7522;)')
    +'</div>','max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','CURRENT STATE')
    +[
      {l:'slope m',v:cur.m.toFixed(4),c:converged?C.green:C.accent},
      {l:'MSE',v:cur.mse.toFixed(4),c:converged?C.green:C.yellow},
      {l:'gradient',v:curGradStr,c:C.orange},
      {l:'steps',v:''+hist.length,c:C.blue},
    ].map(function(row){
      return '<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid '+C.border+';font-size:10px;font-family:monospace;">'
        +'<span style="color:'+C.muted+'">'+row.l+'</span>'
        +'<span style="color:'+row.c+';font-weight:700;">'+row.v+'</span></div>';
    }).join(''),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','LEARNING RATE EFFECTS')
    +[
      {l:'too small',d:'Slow &#8212; many steps needed',c:C.blue},
      {l:'just right',d:'Smooth descent to minimum',c:C.green},
      {l:'too large',d:'Overshoots, may diverge!',c:C.red},
    ].map(function(row){
      return '<div style="display:flex;gap:8px;margin-bottom:6px;">'
        +'<div style="font-size:8px;color:'+row.c+';font-weight:700;font-family:monospace;width:58px;flex-shrink:0;">'+row.l+'</div>'
        +'<div style="font-size:9px;color:'+C.muted+';">'+row.d+'</div></div>';
    }).join(''),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('&#8711;','The Gradient',
    'The gradient tells us the <span style="color:'+C.orange+';font-weight:700;">slope of the cost curve</span> at the current position. '
    +'We always step in the <span style="color:'+C.accent+';font-weight:700;">opposite direction</span> &#8212; downhill toward the minimum. '
    +'Try &#945; &#8776; 0.75 to watch it overshoot!'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 4 — EVALUATION (R²)
═══════════════════════════════════════════ */
function renderEval(){
  var noise=S.noise;
  function rng(s){var x=Math.sin(s)*10000;return x-Math.floor(x);}
  var pts=PTS.map(function(p,i){
    return {x:p.x,y:TM*p.x+TB+(rng(i*7.3+1.1)-0.5)*2*noise};
  });

  var yMean=pts.reduce(function(s,p){return s+p.y;},0)/pts.length;
  var ssTot=pts.reduce(function(s,p){return s+Math.pow(p.y-yMean,2);},0);
  var ssRes=pts.reduce(function(s,p){return s+Math.pow(p.y-(TM*p.x+TB),2);},0);
  var r2=Math.max(0,1-ssRes/ssTot);
  var mse=ssRes/pts.length;
  var rmse=Math.sqrt(mse);
  var r2Col=r2>0.9?C.green:r2>0.7?C.yellow:C.red;

  var sv=plotAxes('Feature (x)','Target (y)');
  /* mean line */
  sv+='<line x1="'+sx(0)+'" y1="'+sy(yMean).toFixed(1)+'" x2="'+sx(10)+'" y2="'+sy(yMean).toFixed(1)+'" stroke="'+C.muted+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>';
  sv+='<text x="'+sx(9.6).toFixed(1)+'" y="'+(sy(yMean)-6).toFixed(1)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">&#563;</text>';
  /* residuals */
  pts.forEach(function(p){
    sv+='<line x1="'+sx(p.x).toFixed(1)+'" y1="'+sy(p.y).toFixed(1)+'" x2="'+sx(p.x).toFixed(1)+'" y2="'+sy(TM*p.x+TB).toFixed(1)+'" stroke="'+C.red+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.55"/>';
  });
  /* line */
  sv+='<line x1="'+sx(0)+'" y1="'+sy(TB).toFixed(1)+'" x2="'+sx(10)+'" y2="'+sy(TM*10+TB).toFixed(1)+'" stroke="'+C.accent+'" stroke-width="2.5"/>';
  /* points */
  pts.forEach(function(p){
    sv+='<circle cx="'+sx(p.x).toFixed(1)+'" cy="'+sy(p.y).toFixed(1)+'" r="4" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  });

  var out=sectionTitle('Model Evaluation','How good is the fit? Quantify it with R&#178;, MSE, and RMSE');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Fit &amp; Residuals')
    +svgBox(sv)
    +sliderRow('noise',noise,0.1,4,0.1,'noise',1)
  );
  out+='</div>';

  /* right */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:10px;','KEY METRICS')
    +[
      {label:'R&#178; Score',value:r2.toFixed(3),desc:'variance explained',color:r2Col},
      {label:'MSE',value:mse.toFixed(3),desc:'Mean Squared Error',color:C.purple},
      {label:'RMSE',value:rmse.toFixed(3),desc:'same units as y',color:C.blue},
    ].map(function(row){
      return '<div style="padding:8px 0;border-bottom:1px solid '+C.border+';display:flex;justify-content:space-between;align-items:center;">'
        +'<div><div style="font-size:10px;font-weight:700;color:'+C.text+';font-family:monospace;">'+row.label+'</div>'
        +'<div style="font-size:8px;color:'+C.muted+';">'+row.desc+'</div></div>'
        +'<div style="font-size:20px;font-weight:800;color:'+row.color+';font-family:monospace;">'+row.value+'</div></div>';
    }).join(''),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','R&#178; DECOMPOSITION')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;font-family:monospace;font-size:9px;text-align:center;">'
    +div('color:'+C.accent+';font-weight:700;line-height:2;','R&#178; = 1 &#8722; SS_res / SS_tot')
    +div('color:'+C.muted+';line-height:1.9;',
      '<span style="color:'+C.red+';">SS_res</span> = '+ssRes.toFixed(1)+'<br>'
      +'<span style="color:'+C.blue+';">SS_tot</span> = '+ssTot.toFixed(1)
    )+'</div>'
    +div('font-size:9px;color:'+C.muted+';margin:8px 0 4px;','R&#178; = '+r2.toFixed(3))
    +'<div style="height:10px;border-radius:5px;background:'+C.border+';overflow:hidden;">'
    +'<div style="height:100%;width:'+(r2*100).toFixed(1)+'%;border-radius:5px;background:'+r2Col+';transition:all .3s;"></div></div>'
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','INTERPRETING R&#178;')
    +[
      {range:'0.9 &#8212; 1.0',qual:'Excellent',c:C.green},
      {range:'0.7 &#8212; 0.9',qual:'Good',c:C.yellow},
      {range:'0.5 &#8212; 0.7',qual:'Moderate',c:C.orange},
      {range:'&lt; 0.5',qual:'Poor',c:C.red},
    ].map(function(row){
      return '<div style="display:flex;justify-content:space-between;padding:3px 0;font-size:9px;font-family:monospace;">'
        +'<span style="color:'+C.muted+';">'+row.range+'</span>'
        +'<span style="color:'+row.c+';font-weight:700;">'+row.qual+'</span></div>';
    }).join(''),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('&#127919;','The Noise Slider',
    'Dial up <span style="color:'+C.yellow+';font-weight:700;">noise</span> to simulate messier real-world data. '
    +'<span style="color:'+C.accent+'">R&#178;</span> drops as variance grows &#8212; the line fits as well as possible, '
    +'but irreducible noise limits how much variance any linear model can explain.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 5 — MULTIPLE REGRESSION
═══════════════════════════════════════════ */
function renderMultiple(){
  var scenarios=[
    {name:'House Prices',color:C.accent,target:'Price ($k)',
      features:['Size (sqft)','Bedrooms','Dist. to city','Age (yrs)','Has garage'],
      weights:[0.18,12.0,-2.5,-0.8,15.0],intercept:80,
      desc:'Predict house sale price from structural and location features.',
      insight:'Garage adds the most value; distance to city has the strongest negative pull.'},
    {name:'Salary',color:C.purple,target:'Salary ($k)',
      features:['Years exp.','Education','Skills score','Industry','Company size'],
      weights:[4.2,8.5,1.1,3.0,0.9],intercept:35,
      desc:'Predict annual salary from professional background.',
      insight:'Education and years of experience dominate salary prediction here.'},
    {name:'Student Score',color:C.yellow,target:'Exam Score',
      features:['Study hrs/day','Sleep hrs','Attendance %','Practice tests','Stress level'],
      weights:[8.0,3.5,0.4,5.5,-6.0],intercept:20,
      desc:'Predict exam scores from student habits and wellbeing.',
      insight:'Stress level has the biggest negative weight. Study hours and sleep both matter.'},
  ];
  var sel=S.mlSel;
  var sc=scenarios[sel];
  var maxA=Math.max.apply(null,sc.weights.map(Math.abs));
  var samp=[2.1,0.5,-0.8,1.2,0.3];
  var pred=sc.intercept+sc.weights.reduce(function(s,w,i){return s+w*samp[i];},0);

  var out=sectionTitle('Multiple Linear Regression','y&#770; = w&#8321;x&#8321; + w&#8322;x&#8322; + &#8230; + w&#8345;x&#8345; + b');
  out+='<div style="display:flex;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:20px;">';
  scenarios.forEach(function(s,i){out+=btnSel(i,sel,s.color,s.name,'mlSel');});
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left: weight bars */
  out+='<div class="card fade" style="flex:1 1 340px;border-color:'+sc.color+';margin-bottom:0;">';
  out+=div('font-size:11px;font-weight:700;color:'+sc.color+';margin-bottom:4px;',sc.name);
  out+=div('font-size:9px;color:'+C.muted+';margin-bottom:14px;',sc.desc);
  out+=div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','FEATURE WEIGHTS');
  sc.features.forEach(function(feat,i){
    var w=sc.weights[i];
    var pct=(Math.abs(w)/maxA)*100;
    var isPos=w>=0;
    out+='<div style="margin-bottom:8px;">'
      +'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
      +'<span style="font-size:9px;color:'+C.text+';font-family:monospace;">'+feat+'</span>'
      +'<span style="font-size:9px;color:'+(isPos?C.green:C.red)+';font-weight:700;font-family:monospace;">'+(isPos?'+':'')+w.toFixed(1)+'</span>'
      +'</div>'
      +'<div style="height:7px;border-radius:4px;background:'+C.border+';overflow:hidden;">'
      +'<div style="width:'+pct.toFixed(1)+'%;height:100%;border-radius:4px;background:'+(isPos?sc.color:C.red)+';opacity:0.75;transition:width .4s;"></div>'
      +'</div></div>';
  });
  out+='<div style="margin-top:12px;padding:8px 12px;background:#08080d;border-radius:8px;border:1px solid '+C.border+';">'
    +div('font-size:8px;color:'+C.muted+';margin-bottom:4px;','INTERCEPT b')
    +div('font-size:14px;font-weight:700;color:'+sc.color+';font-family:monospace;','b = '+sc.intercept)
    +'</div>';
  out+='</div>';

  /* right */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','THE EQUATION')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;text-align:center;font-family:monospace;">'
    +div('font-size:10px;color:'+sc.color+';font-weight:700;line-height:2.2;','y = w&#7488;x + b')
    +div('font-size:8px;color:'+C.muted+';','= &#8721; w&#7522; &middot; x&#7522; + b')
    +'</div>'
    +div('font-size:9px;color:'+C.muted+';line-height:1.8;margin-top:8px;',
      'Weight vector <span style="color:'+sc.color+'">w</span> &middot; feature vector <span style="color:'+C.blue+'">x</span> + bias <span style="color:'+C.purple+'">b</span>.'
    ),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','EXAMPLE PREDICTION')
    +sc.features.map(function(f,i){
      return '<div style="display:flex;justify-content:space-between;font-size:9px;font-family:monospace;padding:2px 0;">'
        +'<span style="color:'+C.muted+'">'+f+'</span>'
        +'<span style="color:'+C.blue+'">'+samp[i]+'</span></div>';
    }).join('')
    +'<div style="margin-top:10px;padding:8px 12px;background:'+hex(sc.color,.1)+';border-radius:8px;border:1px solid '+hex(sc.color,.3)+';text-align:center;">'
    +div('font-size:8px;color:'+C.muted+';margin-bottom:2px;','PREDICTED '+sc.target.toUpperCase())
    +div('font-size:22px;font-weight:800;color:'+sc.color+';font-family:monospace;',pred.toFixed(1))
    +'</div>','max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','KEY ASSUMPTIONS')
    +[
      {l:'Linearity',c:C.green},
      {l:'Independent errors',c:C.green},
      {l:'Homoscedasticity',c:C.yellow},
      {l:'Normal errors',c:C.blue},
      {l:'No multicollinearity',c:C.orange},
    ].map(function(a){
      return '<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:9px;font-family:monospace;">'
        +'<div style="width:8px;height:8px;border-radius:2px;background:'+a.c+';flex-shrink:0;"></div>'
        +'<span style="color:'+C.muted+'">'+a.l+'</span></div>';
    }).join(''),'max-width:none;margin:0;'
  );
  out+='</div></div>';

  /* 1D → nD progression */
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','From 1D to nD')
    +'<div style="display:flex;gap:0;align-items:center;flex-wrap:wrap;justify-content:center;">'
    +[
      {label:'Simple',sub:'y = mx + b',dim:'1 feature',c:C.accent},
      null,
      {label:'Multiple',sub:'y = w&#8321;x&#8321;+w&#8322;x&#8322;+b',dim:'2 features',c:C.purple},
      null,
      {label:'General',sub:'y = w&#7488;x + b',dim:'n features',c:C.yellow},
    ].map(function(item){
      if(!item) return '<div style="font-size:20px;color:'+C.dim+';padding:0 8px;">&#8594;</div>';
      return '<div style="text-align:center;padding:10px 16px;background:#08080d;border-radius:8px;border:1px solid '+hex(item.c,.3)+';margin:4px;">'
        +'<div style="font-size:11px;font-weight:800;color:'+item.c+';">'+item.label+'</div>'
        +'<div style="font-size:9px;color:'+C.muted+';font-family:monospace;margin:4px 0;">'+item.sub+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';">'+item.dim+'</div></div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#128161;','Weights = Importance',
    'A <span style="color:'+C.green+';font-weight:700;">large positive weight</span> strongly pushes predictions up; '
    +'a <span style="color:'+C.red+';font-weight:700;">large negative weight</span> pulls them down. '
    +'Weights are only comparable when features share the <span style="color:'+sc.color+';font-weight:700;">same scale</span> &#8212; always standardize first!'
  );
  return out;
}

/* ═══════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════ */
var TABS=[
  '&#128200; Intro &amp; Fitting',
  '&#128201; Cost Function',
  '&#8711; Gradient Descent',
  '&#128202; Evaluation (R&#178;)',
  '&#9993; Multiple Regression'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.blue+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Linear Regression</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough &#8212; from fitting a line to gradient descent to R&#178;')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  if(S.tab===0) html+=renderIntro();
  else if(S.tab===1) html+=renderCost();
  else if(S.tab===2) html+=renderGradient();
  else if(S.tab===3) html+=renderEval();
  else if(S.tab===4) html+=renderMultiple();
  html+='</div>';
  return html;
}

/* ─── GRADIENT DESCENT STEP ─── */
function doGdStep(){
  if(S.gdHist.length>=50){
    S.gdRunning=false;
    clearInterval(S.gdTimer);
    S.gdTimer=null;
    render();
    return;
  }
  var last=S.gdHist[S.gdHist.length-1];
  var g=gradM(last.m,TB);
  var nm=last.m-S.gdLr*g;
  S.gdHist=S.gdHist.concat([{m:nm,mse:mseAt(nm,TB),grad:g}]);
  render();
}

/* ─── RENDER + BIND ─── */
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
        if(action==='tab'){
          if(S.gdRunning){S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;}
          S.tab=idx;
          render();
        }
        else if(action==='gdRun'){
          S.gdRunning=!S.gdRunning;
          if(S.gdRunning){
            S.gdTimer=setInterval(doGdStep,280);
          } else {
            clearInterval(S.gdTimer);S.gdTimer=null;
          }
          render();
        }
        else if(action==='gdStep'){
          if(!S.gdRunning) doGdStep();
        }
        else if(action==='gdReset'){
          S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;
          S.gdHist=[{m:3.2,mse:mseAt(3.2,TB)}];
          render();
        }
        else if(action==='mlSel'){S.mlSel=idx;render();}
      });
    }
    else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='slope'){S.slope=val;render();}
        else if(action==='intercept'){S.intercept=val;render();}
        else if(action==='costSlope'){S.costSlope=val;render();}
        else if(action==='noise'){S.noise=val;render();}
        else if(action==='gdLr'){
          S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;
          S.gdHist=[{m:3.2,mse:mseAt(3.2,TB)}];
          S.gdLr=val;
          render();
        }
      });
    }
  });
}

render();
</script>
</body>
</html>"""

LR_VISUAL_HEIGHT = 1100

