"""
Self-contained HTML visual for Logistic Regression.
5 interactive tabs: Sigmoid Function, Decision Boundary, BCE Loss,
Gradient Descent, Softmax (Multi-Class).
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(LOR_VISUAL_HTML, height=LOR_VISUAL_HEIGHT, scrolling=True)
"""

LOR_VISUAL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#f472b6;cursor:pointer;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .3s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#f472b6;color:#f472b6;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
.section-title p{font-size:11px;color:#71717a;}
.insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(244,114,182,.06);border-radius:10px;border:1px solid rgba(244,114,182,.2);}
.ins-title{font-size:11px;font-weight:700;color:#f472b6;margin-bottom:6px;}
.ins-body{font-size:11px;color:#71717a;line-height:1.8;}
</style>
</head>
<body>
<div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
<script>
/* ─── PALETTE ─── */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
  accent:"#f472b6",blue:"#38bdf8",purple:"#a78bfa",
  yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
  dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
  orange:"#fb923c",teal:"#2dd4bf"};

/* ─── MATH ─── */
function sigmoid(z){
  if(z>500) return 1.0;
  if(z<-500) return 0.0;
  return 1/(1+Math.exp(-z));
}

/* ─── BINARY DATASET (2D, linearly separable) ─── */
var PTS=[
  {x1:4.5,x2:4.8,y:1},{x1:5.2,x2:3.9,y:1},{x1:3.8,x2:5.2,y:1},
  {x1:5.8,x2:4.5,y:1},{x1:4.2,x2:5.6,y:1},{x1:6.0,x2:3.5,y:1},
  {x1:1.5,x2:1.8,y:0},{x1:2.2,x2:0.9,y:0},{x1:0.8,x2:2.5,y:0},
  {x1:2.8,x2:1.5,y:0},{x1:1.2,x2:2.8,y:0},{x1:2.0,x2:0.5,y:0}
];

/* ─── LOSS / ACCURACY ─── */
function bceLoss(pts,w1,w2,b){
  var s=0,eps=1e-10;
  pts.forEach(function(p){
    var yh=sigmoid(w1*p.x1+w2*p.x2+b);
    s+=-(p.y*Math.log(yh+eps)+(1-p.y)*Math.log(1-yh+eps));
  });
  return s/pts.length;
}
function accuracy(pts,w1,w2,b){
  var c=0;
  pts.forEach(function(p){
    if((sigmoid(w1*p.x1+w2*p.x2+b)>=0.5?1:0)===p.y) c++;
  });
  return c/pts.length;
}

/* ─── 1-D GD (fix w2=1, b=−3.5, sweep w1) ─── */
function bce1d(w){return bceLoss(PTS,w,1.0,-3.5);}
function grad1d(w){
  var g=0;
  PTS.forEach(function(p){g+=(sigmoid(w*p.x1+1.0*p.x2-3.5)-p.y)*p.x1;});
  return g/PTS.length;
}
/* numeric minimum on [−1,5] */
var _bestW=1.0,_bestL=bce1d(1.0);
(function(){for(var v=-1;v<=5;v+=0.02){var l=bce1d(v);if(l<_bestL){_bestL=l;_bestW=v;}}})();

/* ─── STATE ─── */
var S={
  tab:0,
  sigZ:2.0,
  dbW1:1.0,dbW2:1.0,dbB:-3.5,
  bceYhat:0.8,
  gdHist:[{w:3.5,bce:bce1d(3.5)}],
  gdLr:0.3,gdRunning:false,gdTimer:null,
  smZ:[2.0,1.0,0.5]
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
function sliderRow(action,val,min,max,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:48px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ═══════════════════════════════════════════
   TAB 1 — SIGMOID FUNCTION
═══════════════════════════════════════════ */
function renderSigmoid(){
  var z0=S.sigZ;
  var yhat=sigmoid(z0);
  var pClass=yhat>=0.5?1:0;
  var confCol=yhat>=0.5?C.green:C.red;
  if(Math.abs(yhat-0.5)<0.05) confCol=C.yellow;

  var VW=440,VH=270,PL=50,PR=16,PT=16,PB=38;
  var PW=VW-PL-PR,PH=VH-PT-PB;
  var zMin=-7,zMax=7;
  function sx(z){return PL+((z-zMin)/(zMax-zMin))*PW;}
  function sy(p){return PT+PH*(1-p);}

  var sv='';
  /* grid */
  [-6,-4,-2,0,2,4,6].forEach(function(v){
    sv+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  [0,0.25,0.5,0.75,1].forEach(function(v){
    sv+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  sv+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [-6,-4,-2,0,2,4,6].forEach(function(v){
    sv+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  [0,0.25,0.5,0.75,1].forEach(function(v){
    sv+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v.toFixed(2)+'</text>';
  });
  sv+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">z  (linear score = w·x + b)</text>';
  sv+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">σ(z)</text>';

  /* threshold line */
  sv+='<line x1="'+PL+'" y1="'+sy(0.5).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(0.5).toFixed(1)+'" stroke="'+C.yellow+'" stroke-width="1" stroke-dasharray="5,3" opacity="0.6"/>';
  sv+='<text x="'+(PL+PW-4)+'" y="'+(sy(0.5)-5).toFixed(1)+'" text-anchor="end" fill="'+C.yellow+'" font-size="7.5" font-family="monospace">threshold = 0.5</text>';

  /* sigmoid curve */
  var pts=[];
  for(var z=zMin;z<=zMax+0.01;z+=0.1) pts.push(sx(z).toFixed(1)+','+sy(sigmoid(z)).toFixed(1));
  sv+='<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+C.accent+'" stroke-width="2.5"/>';

  /* readout lines */
  sv+='<line x1="'+sx(z0).toFixed(1)+'" y1="'+PT+'" x2="'+sx(z0).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+confCol+'" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>';
  sv+='<line x1="'+PL+'" y1="'+sy(yhat).toFixed(1)+'" x2="'+sx(z0).toFixed(1)+'" y2="'+sy(yhat).toFixed(1)+'" stroke="'+confCol+'" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>';
  sv+='<circle cx="'+sx(z0).toFixed(1)+'" cy="'+sy(yhat).toFixed(1)+'" r="7" fill="'+confCol+'" stroke="#0a0a0f" stroke-width="2"/>';
  var labelX=z0<3?sx(z0)+12:sx(z0)-12;
  var labelAnchor=z0<3?'start':'end';
  sv+='<text x="'+labelX.toFixed(1)+'" y="'+(sy(yhat)-10).toFixed(1)+'" text-anchor="'+labelAnchor+'" fill="'+confCol+'" font-size="9" font-family="monospace" font-weight="700">σ('+z0.toFixed(1)+') = '+yhat.toFixed(3)+'</text>';

  /* formula box */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="168" height="21" rx="4" fill="#0a0a0f" opacity="0.88"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+16)+'" fill="'+C.accent+'" font-size="9.5" font-family="monospace" font-weight="700">σ(z) = 1 / (1 + e&#8315;&#7611;)</text>';

  var out=sectionTitle('The Sigmoid Function','Squashes any real number z into a probability ∈ (0, 1)');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* ── left ── */
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Interactive Sigmoid')
    +svgBox(sv,VW,VH)
    +sliderRow('sigZ',z0,-6,6,0.1,'z score',1)
  );
  /* pipeline */
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Prediction Pipeline')
    +'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:center;">'
    +'<div style="text-align:center;padding:8px 10px;background:#08080d;border-radius:8px;border:1px solid '+C.border+';">'
    +div('font-size:8px;color:'+C.muted+';','INPUT')
    +div('font-size:13px;font-weight:800;color:'+C.blue+';font-family:monospace;','x, w, b')+'</div>'
    +'<div style="font-size:18px;color:'+C.dim+';">→</div>'
    +'<div style="text-align:center;padding:8px 10px;background:#08080d;border-radius:8px;border:1px solid '+hex(C.purple,.4)+';">'
    +div('font-size:8px;color:'+C.muted+';','LINEAR SCORE')
    +div('font-size:13px;font-weight:800;color:'+C.purple+';font-family:monospace;','z = '+z0.toFixed(2))+'</div>'
    +'<div style="font-size:18px;color:'+C.dim+';">→</div>'
    +'<div style="text-align:center;padding:8px 10px;background:#08080d;border-radius:8px;border:1px solid '+hex(C.accent,.4)+';">'
    +div('font-size:8px;color:'+C.muted+';','SIGMOID')
    +div('font-size:13px;font-weight:800;color:'+C.accent+';font-family:monospace;','ŷ = '+yhat.toFixed(3))+'</div>'
    +'<div style="font-size:18px;color:'+C.dim+';">→</div>'
    +'<div style="text-align:center;padding:8px 10px;background:'+hex(confCol,0.1)+';border-radius:8px;border:1.5px solid '+hex(confCol,.5)+';">'
    +div('font-size:8px;color:'+C.muted+';','PREDICT')
    +div('font-size:18px;font-weight:800;color:'+confCol+';font-family:monospace;','class '+pClass)+'</div>'
    +'</div>'
  );
  out+='</div>';

  /* ── right ── */
  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','THE FORMULA')
    +div('font-size:17px;font-weight:800;color:'+C.accent+';font-family:monospace;text-align:center;margin:8px 0;','σ(z) = 1 / (1+e&#8315;&#7611;)')
    +div('font-size:9px;color:'+C.muted+';line-height:1.9;',
      '<div><span style="color:'+C.accent+';font-weight:700;">σ(z)</span> — probability of class 1</div>'
      +'<div><span style="color:'+C.purple+';font-weight:700;">z</span> — linear score = w·x + b</div>'
      +'<div style="margin-top:8px;border-top:1px solid '+C.border+';padding-top:8px;">'
      +'<span style="color:'+C.teal+';font-weight:700;">Derivative:</span><br>'
      +'σ\'(z) = σ(z) · (1 − σ(z))</div>'
    ),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','KEY VALUES')
    +[
      {z:-6,val:'0.002',label:'confident: class 0',c:C.red},
      {z:-2,val:'0.119',label:'leaning class 0',c:C.orange},
      {z:0, val:'0.500',label:'maximally uncertain',c:C.yellow},
      {z:2, val:'0.881',label:'leaning class 1',c:C.teal},
      {z:6, val:'0.998',label:'confident: class 1',c:C.green},
    ].map(function(r){
      return '<div style="display:flex;align-items:center;justify-content:space-between;padding:3px 0;border-bottom:1px solid '+C.border+';font-size:9px;font-family:monospace;">'
        +'<span style="color:'+C.muted+';width:28px;">z='+r.z+'</span>'
        +'<span style="color:'+r.c+';font-weight:700;width:40px;">'+r.val+'</span>'
        +'<span style="color:'+C.dim+';font-size:7.5px;">'+r.label+'</span></div>';
    }).join(''),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','SYMMETRY')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;text-align:center;font-family:monospace;margin-bottom:6px;">'
    +div('font-size:10px;color:'+C.purple+';font-weight:700;line-height:2.2;','σ(−z) = 1 − σ(z)')
    +'</div>'
    +div('font-size:8.5px;color:'+C.muted+';','Symmetric around (0, 0.5). Step function ≡ sign(z), but non-differentiable.')
    ,'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('💡','Why Sigmoid?',
    'The sigmoid converts an unbounded linear score into a probability. Unlike a step function, it is '
    +'<span style="color:'+C.teal+';font-weight:700;">differentiable everywhere</span> — '
    +'and its derivative <span style="color:'+C.accent+';font-weight:700;">σ\'(z) = σ(z)·(1−σ(z))</span> '
    +'is expressible in terms of its own output, making backpropagation cheap. Every output neuron in a binary classification network <em>is</em> a sigmoid.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 2 — DECISION BOUNDARY
═══════════════════════════════════════════ */
function renderDecisionBoundary(){
  var w1=S.dbW1,w2=S.dbW2,b=S.dbB;
  var acc=accuracy(PTS,w1,w2,b);
  var loss=bceLoss(PTS,w1,w2,b);
  var accCol=acc>=0.95?C.green:acc>=0.7?C.yellow:C.red;
  var lossCol=loss<0.3?C.green:loss<0.8?C.yellow:C.red;

  var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
  var PW=VW-PL-PR,PH=VH-PT-PB;
  var xMin=0,xMax=7.5,yMin=0,yMax=7;
  function sx(x){return PL+((x-xMin)/(xMax-xMin))*PW;}
  function sy(y){return PT+PH-((y-yMin)/(yMax-yMin))*PH;}

  var sv='';
  /* grid */
  [0,1,2,3,4,5,6,7].forEach(function(v){
    sv+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
  });
  sv+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,2,4,6].forEach(function(v){
    sv+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
    sv+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  sv+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">x&#8321;</text>';
  sv+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">x&#8322;</text>';

  /* probability shading */
  for(var xi=0.3;xi<=7.4;xi+=0.65){
    for(var yi=0.3;yi<=6.8;yi+=0.6){
      var z=w1*xi+w2*yi+b;
      var p=sigmoid(z);
      var col=p>=0.5?C.blue:C.orange;
      var al=(Math.abs(p-0.5)*0.55).toFixed(2);
      sv+='<circle cx="'+sx(xi).toFixed(1)+'" cy="'+sy(yi).toFixed(1)+'" r="15" fill="'+col+'" opacity="'+al+'"/>';
    }
  }

  /* decision boundary line: w1*x1 + w2*x2 + b = 0 */
  if(Math.abs(w2)>0.05){
    /* compute x2 at x1=xMin and x1=xMax */
    var rawPts=[[xMin,-(w1*xMin+b)/w2],[xMax,-(w1*xMax+b)/w2]];
    /* clip each point */
    var clipped=[];
    rawPts.forEach(function(pt){
      var x=pt[0],y=pt[1];
      if(y<yMin){
        x=xMin+(yMin-rawPts[0][1])*(xMax-xMin)/(rawPts[1][1]-rawPts[0][1]+1e-10);
        y=yMin;
      }
      if(y>yMax){
        x=xMin+(yMax-rawPts[0][1])*(xMax-xMin)/(rawPts[1][1]-rawPts[0][1]+1e-10);
        y=yMax;
      }
      if(x>=xMin-0.1&&x<=xMax+0.1&&y>=yMin-0.1&&y<=yMax+0.1) clipped.push([x,y]);
    });
    if(clipped.length>=2){
      sv+='<line x1="'+sx(clipped[0][0]).toFixed(1)+'" y1="'+sy(clipped[0][1]).toFixed(1)+'" x2="'+sx(clipped[1][0]).toFixed(1)+'" y2="'+sy(clipped[1][1]).toFixed(1)+'" stroke="'+C.accent+'" stroke-width="2.5"/>';
      var mx=(clipped[0][0]+clipped[1][0])/2,my=(clipped[0][1]+clipped[1][1])/2;
      if(mx>=xMin&&mx<=xMax&&my>=yMin&&my<=yMax)
        sv+='<text x="'+sx(mx).toFixed(1)+'" y="'+(sy(my)-8).toFixed(1)+'" text-anchor="middle" fill="'+C.accent+'" font-size="7.5" font-family="monospace" font-weight="700">w&#xB7;x + b = 0</text>';
    }
  } else if(Math.abs(w1)>0.05){
    /* vertical-ish boundary */
    var bx=-(b+w2*3.5)/w1;
    if(bx>=xMin&&bx<=xMax)
      sv+='<line x1="'+sx(bx).toFixed(1)+'" y1="'+PT+'" x2="'+sx(bx).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="2.5"/>';
  }

  /* data points */
  PTS.forEach(function(p){
    var pred=sigmoid(w1*p.x1+w2*p.x2+b)>=0.5?1:0;
    var correct=pred===p.y;
    var fillCol=p.y===1?C.blue:C.orange;
    sv+='<circle cx="'+sx(p.x1).toFixed(1)+'" cy="'+sy(p.x2).toFixed(1)+'" r="5.5" fill="'+fillCol+'" stroke="'+(correct?'#0a0a0f':C.red)+'" stroke-width="'+(correct?1.5:2.5)+'"/>';
    if(!correct) sv+='<text x="'+sx(p.x1).toFixed(1)+'" y="'+(sy(p.x2)+3).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="8" font-weight="700">&#10007;</text>';
  });

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="126" height="38" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<circle cx="'+(PL+14)+'" cy="'+(PT+14)+'" r="4" fill="'+C.blue+'"/>';
  sv+='<text x="'+(PL+22)+'" y="'+(PT+18)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">Class 1 (y=1)</text>';
  sv+='<circle cx="'+(PL+14)+'" cy="'+(PT+30)+'" r="4" fill="'+C.orange+'"/>';
  sv+='<text x="'+(PL+22)+'" y="'+(PT+34)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">Class 0 (y=0)</text>';

  var out=sectionTitle('Decision Boundary','The boundary is always linear: w₁x₁ + w₂x₂ + b = 0');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Adjust the Decision Boundary')
    +svgBox(sv,VW,VH)
    +sliderRow('dbW1',w1,-2,4,0.1,'weight w&#8321;',1)
    +sliderRow('dbW2',w2,-2,4,0.1,'weight w&#8322;',1)
    +sliderRow('dbB',b,-6,2,0.1,'bias b',1)
  );
  out+='</div>';

  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','LIVE METRICS')
    +'<div style="text-align:center;margin:8px 0;">'
    +div('font-size:11px;color:'+C.muted+';margin-bottom:4px;','Accuracy')
    +div('font-size:30px;font-weight:800;color:'+accCol+';transition:color .3s;',(acc*100).toFixed(0)+'%')
    +'</div>'
    +'<div style="height:8px;border-radius:4px;background:'+C.border+';overflow:hidden;margin-bottom:12px;">'
    +'<div style="height:100%;width:'+(acc*100).toFixed(1)+'%;border-radius:4px;background:'+accCol+';transition:all .3s;"></div></div>'
    +div('font-size:9px;color:'+C.muted+';margin-bottom:4px;','BCE Loss')
    +div('font-size:22px;font-weight:800;color:'+lossCol+';font-family:monospace;',loss.toFixed(3))
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','BOUNDARY EQUATION')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;text-align:center;font-family:monospace;">'
    +div('font-size:9.5px;color:'+C.accent+';font-weight:700;line-height:2.2;',
      w1.toFixed(1)+'x&#8321; + '+w2.toFixed(1)+'x&#8322; + ('+b.toFixed(1)+') = 0'
    )
    +(Math.abs(w2)>0.1?div('font-size:8px;color:'+C.muted+';margin-top:4px;',
      'x&#8322; = '+(-(w1/w2)).toFixed(2)+'x&#8321; + '+(-(b/w2)).toFixed(2)
    ):'')
    +'</div>','max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','COLOUR KEY')
    +div('font-size:9px;color:'+C.muted+';line-height:2.0;',
      '<span style="color:'+C.blue+';">■</span> Blue region → P(y=1) &gt; 0.5<br>'
      +'<span style="color:'+C.orange+';">■</span> Orange region → P(y=0) &gt; 0.5<br>'
      +'<span style="color:'+C.accent+';">—</span> Boundary: σ(z) = 0.5 exactly<br>'
      +'<span style="color:'+C.red+';">✗</span> Red ring = misclassified point'
    ),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('📐','Always Linear',
    'No matter what weights you choose, the decision boundary is always a '
    +'<span style="color:'+C.accent+';font-weight:700;">straight line</span> (hyperplane in p dimensions). '
    +'Rotate and shift it with the sliders. If the data isn\'t linearly separable, logistic regression finds '
    +'the <span style="color:'+C.yellow+';font-weight:700;">best possible linear boundary</span> but cannot reach 100% accuracy — '
    +'you need feature engineering, kernels, or a neural network for curved boundaries.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 3 — BINARY CROSS-ENTROPY LOSS
═══════════════════════════════════════════ */
function renderLoss(){
  var yhat=S.bceYhat;
  var bce=-Math.log(yhat+1e-12);
  var mse=Math.pow(1-yhat,2);
  var bceCol=bce<0.5?C.green:bce<1.5?C.yellow:C.red;

  var VW=440,VH=255,PL=52,PR=16,PT=16,PB=38;
  var PW=VW-PL-PR,PH=VH-PT-PB;
  var lMax=4;
  function sx(y){return PL+(y)*PW;}
  function sy(l){return PT+PH*(1-Math.min(1,l/lMax));}

  var sv='';
  /* grid */
  [0,0.2,0.4,0.6,0.8,1].forEach(function(v){
    sv+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v.toFixed(1)+'</text>';
  });
  [0,1,2,3,4].forEach(function(v){
    sv+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  sv+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">predicted probability ŷ  (true label y = 1)</text>';
  sv+='<text x="12" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,12,'+(PT+PH/2)+')">loss</text>';

  /* BCE curve */
  var bcePts=[];
  for(var p=0.01;p<=0.99;p+=0.01){var l=-Math.log(p);if(l<=lMax) bcePts.push(sx(p).toFixed(1)+','+sy(l).toFixed(1));}
  sv+='<polyline points="'+bcePts.join(' ')+'" fill="none" stroke="'+C.accent+'" stroke-width="2.5"/>';
  sv+='<text x="'+sx(0.09)+'" y="'+(sy(2.4)-6).toFixed(1)+'" fill="'+C.accent+'" font-size="8.5" font-family="monospace" font-weight="700">BCE = −log(ŷ)</text>';

  /* MSE curve */
  var msePts=[];
  for(var p=0;p<=1;p+=0.01) msePts.push(sx(p).toFixed(1)+','+sy(Math.pow(1-p,2)).toFixed(1));
  sv+='<polyline points="'+msePts.join(' ')+'" fill="none" stroke="'+C.blue+'" stroke-width="2" stroke-dasharray="6,3" opacity="0.8"/>';
  sv+='<text x="'+sx(0.08)+'" y="'+(sy(0.82)-6).toFixed(1)+'" fill="'+C.blue+'" font-size="8.5" font-family="monospace">MSE = (1−ŷ)²</text>';

  /* interactive readout */
  sv+='<line x1="'+sx(yhat).toFixed(1)+'" y1="'+PT+'" x2="'+sx(yhat).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+bceCol+'" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>';
  if(bce<=lMax) sv+='<circle cx="'+sx(yhat).toFixed(1)+'" cy="'+sy(bce).toFixed(1)+'" r="7" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="2"/>';
  sv+='<circle cx="'+sx(yhat).toFixed(1)+'" cy="'+sy(mse).toFixed(1)+'" r="5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="2"/>';
  var lx=(yhat<0.7)?sx(yhat)+10:sx(yhat)-10;
  var la=(yhat<0.7)?'start':'end';
  if(bce<=lMax) sv+='<text x="'+lx.toFixed(1)+'" y="'+(sy(bce)-8).toFixed(1)+'" text-anchor="'+la+'" fill="'+C.accent+'" font-size="9" font-family="monospace" font-weight="700">'+bce.toFixed(3)+'</text>';
  sv+='<text x="'+lx.toFixed(1)+'" y="'+(sy(mse)+18).toFixed(1)+'" text-anchor="'+la+'" fill="'+C.blue+'" font-size="9" font-family="monospace">'+mse.toFixed(4)+'</text>';

  /* comparison table */
  var rows=[
    [0.95,(-Math.log(0.95)).toFixed(3),(Math.pow(0.05,2)).toFixed(4),'confident, correct'],
    [0.70,(-Math.log(0.70)).toFixed(3),(Math.pow(0.30,2)).toFixed(4),'moderate'],
    [0.50,(-Math.log(0.50)).toFixed(3),(Math.pow(0.50,2)).toFixed(4),'uncertain'],
    [0.20,(-Math.log(0.20)).toFixed(3),(Math.pow(0.80,2)).toFixed(4),'leaning wrong'],
    [0.05,(-Math.log(0.05)).toFixed(3),(Math.pow(0.95,2)).toFixed(4),'confident, WRONG'],
  ];

  var out=sectionTitle('Binary Cross-Entropy Loss','BCE screams loud when you\'re confidently wrong — MSE barely notices');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Loss vs Predicted Probability  (y = 1)')
    +svgBox(sv,VW,VH)
    +sliderRow('bceYhat',yhat,0.01,0.99,0.01,'ŷ  (pred.)',2)
  );
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','BCE vs MSE Side-by-Side  (y = 1)')
    +'<div style="display:grid;grid-template-columns:0.8fr 1fr 1fr 1.3fr;gap:2px;font-size:9px;font-family:monospace;">'
    +'<div style="color:'+C.muted+';padding:4px 2px;">ŷ</div>'
    +'<div style="color:'+C.accent+';padding:4px 2px;text-align:right;">BCE</div>'
    +'<div style="color:'+C.blue+';padding:4px 2px;text-align:right;">MSE</div>'
    +'<div style="color:'+C.muted+';padding:4px 2px;text-align:right;">note</div>'
    +rows.map(function(r){
      var col=r[0]>=0.5?C.green:C.red;
      return '<div style="color:'+col+';padding:3px 2px;border-top:1px solid '+C.border+';">'+r[0]+'</div>'
        +'<div style="color:'+C.accent+';padding:3px 2px;border-top:1px solid '+C.border+';text-align:right;">'+r[1]+'</div>'
        +'<div style="color:'+C.blue+';padding:3px 2px;border-top:1px solid '+C.border+';text-align:right;">'+r[2]+'</div>'
        +'<div style="color:'+C.dim+';padding:3px 2px;border-top:1px solid '+C.border+';text-align:right;font-size:8px;">'+r[3]+'</div>';
    }).join('')+'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','CURRENT  (y = 1)')
    +'<div style="text-align:center;margin:8px 0;">'
    +div('font-size:11px;color:'+C.muted+';','BCE at ŷ = '+yhat.toFixed(2))
    +div('font-size:32px;font-weight:800;color:'+bceCol+';font-family:monospace;transition:all .3s;',bce<=lMax?bce.toFixed(3):'∞')
    +'</div>'
    +div('font-size:9px;color:'+C.muted+';','MSE: <span style="color:'+C.blue+';font-weight:700;">'+mse.toFixed(4)+'</span>')
    +div('font-size:9px;color:'+C.dim+';margin-top:4px;','BCE / MSE ratio: <span style="color:'+C.yellow+';">×'+(bce/(mse+1e-10)<999?(bce/(mse+1e-10)).toFixed(1):'∞')+'</span>')
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','THE FORMULA')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;text-align:center;font-family:monospace;">'
    +div('font-size:9px;color:'+C.accent+';font-weight:700;line-height:2.2;','L = −[y log(ŷ) + (1−y) log(1−ŷ)]')
    +div('font-size:8px;color:'+C.muted+';line-height:1.9;',
      'y=1: loss = −log(ŷ)<br>y=0: loss = −log(1−ŷ)'
    )+'</div>'
    +div('font-size:9px;color:'+C.muted+';margin-top:8px;','MLE under <span style="color:'+C.teal+';font-weight:700;">Bernoulli</span> dist.<br><span style="font-size:8px;">same way MSE is MLE under Gaussian</span>')
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','CONVEXITY')
    +div('font-size:9px;color:'+C.muted+';line-height:2.0;',
      '✓ <span style="color:'+C.green+';font-weight:700;">BCE + sigmoid</span> — convex<br>'
      +'✗ <span style="color:'+C.red+';font-weight:700;">MSE + sigmoid</span> — NOT convex<br>'
      +'<span style="font-size:8px;color:'+C.dim+';">Convex ⇒ gradient descent always reaches the global minimum</span>'
    ),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('⚡','Why BCE and Not MSE?',
    'When ŷ → 0 but y = 1, <span style="color:'+C.accent+';font-weight:700;">BCE → ∞</span>. '
    +'MSE tops out at 1. That massive penalty forces the model to never be '
    +'<span style="color:'+C.red+';font-weight:700;">confidently wrong</span>, '
    +'producing better-calibrated probabilities. BCE also makes the loss surface convex — '
    +'MSE with sigmoid activations creates local minima that trap gradient descent.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 4 — GRADIENT DESCENT
═══════════════════════════════════════════ */
function renderGradient(){
  var hist=S.gdHist;
  var cur=hist[hist.length-1];
  var step=hist.length-1;

  var VW=440,VH=235,PL=52,PR=16,PT=16,PB=38;
  var PW=VW-PL-PR,PH=VH-PT-PB;
  var wMin=-1,wMax=5,cMax=1.4;
  function sx(w){return PL+((w-wMin)/(wMax-wMin))*PW;}
  function sy(c){return PT+PH*(1-Math.min(1,c/cMax));}

  var sv='';
  /* grid */
  [-1,0,1,2,3,4,5].forEach(function(v){
    sv+='<line x1="'+sx(v).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+sx(v).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  [0,0.5,1.0,1.4].forEach(function(v){
    sv+='<line x1="'+PL+'" y1="'+sy(v).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+(PL-6)+'" y="'+(sy(v)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v.toFixed(1)+'</text>';
  });
  sv+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">weight w&#8321;  (w&#8322;=1, b=&#8722;3.5 fixed)</text>';
  sv+='<text x="12" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,12,'+(PT+PH/2)+')">BCE loss</text>';

  /* loss curve */
  var cPts=[];
  for(var w=wMin;w<=wMax+0.01;w+=0.06){var c=bce1d(w);if(c<=cMax) cPts.push(sx(w).toFixed(1)+','+sy(c).toFixed(1));}
  sv+='<polyline points="'+cPts.join(' ')+'" fill="none" stroke="'+C.purple+'" stroke-width="2.5"/>';

  /* global minimum */
  sv+='<circle cx="'+sx(_bestW).toFixed(1)+'" cy="'+sy(_bestL).toFixed(1)+'" r="6" fill="'+C.green+'" opacity="0.9"/>';
  sv+='<text x="'+(sx(_bestW)+10).toFixed(1)+'" y="'+(sy(_bestL)-8).toFixed(1)+'" fill="'+C.green+'" font-size="8" font-family="monospace" font-weight="700">global minimum</text>';

  /* gradient path (last 8 steps) */
  var nShow=Math.min(8,hist.length);
  for(var i=hist.length-nShow;i<hist.length-1;i++){
    var h0=hist[i],h1=hist[i+1];
    var alpha=(0.3+(i-(hist.length-nShow))/(nShow)*0.7).toFixed(2);
    sv+='<line x1="'+sx(h0.w).toFixed(1)+'" y1="'+sy(Math.min(cMax,h0.bce)).toFixed(1)+'" x2="'+sx(h1.w).toFixed(1)+'" y2="'+sy(Math.min(cMax,h1.bce)).toFixed(1)+'" stroke="'+C.accent+'" stroke-width="1.8" opacity="'+alpha+'"/>';
    sv+='<circle cx="'+sx(h0.w).toFixed(1)+'" cy="'+sy(Math.min(cMax,h0.bce)).toFixed(1)+'" r="3" fill="'+C.accent+'" opacity="'+alpha+'"/>';
  }

  /* current position */
  sv+='<line x1="'+sx(cur.w).toFixed(1)+'" y1="'+PT+'" x2="'+sx(cur.w).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>';
  sv+='<circle cx="'+sx(cur.w).toFixed(1)+'" cy="'+sy(Math.min(cMax,cur.bce)).toFixed(1)+'" r="7" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<text x="'+(PL+6)+'" y="'+(PT+13)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">BCE='+cur.bce.toFixed(4)+'   w='+cur.w.toFixed(3)+'</text>';

  /* loss history sparkline */
  var LH=70,LW=380,LL=20,LT=8;
  var maxH=Math.max.apply(null,hist.map(function(h){return h.bce;}));
  var minH=Math.min.apply(null,hist.map(function(h){return h.bce;}));
  var hPts=hist.map(function(h,i){
    var hx=LL+(i/(Math.max(1,hist.length-1)))*(LW-LL);
    var hy=LT+(LH-LT-4)*(1-(h.bce-minH)/(maxH-minH+1e-10));
    return hx.toFixed(1)+','+hy.toFixed(1);
  });
  var lhSvg='<line x1="'+LL+'" y1="'+LT+'" x2="'+LL+'" y2="'+LH+'" stroke="'+C.dim+'" stroke-width="1"/>';
  lhSvg+='<line x1="'+LL+'" y1="'+LH+'" x2="'+LW+'" y2="'+LH+'" stroke="'+C.dim+'" stroke-width="1"/>';
  lhSvg+='<text x="'+LL+'" y="'+(LH+14)+'" fill="'+C.muted+'" font-size="7" font-family="monospace">step 0</text>';
  lhSvg+='<text x="'+LW+'" y="'+(LH+14)+'" text-anchor="end" fill="'+C.muted+'" font-size="7" font-family="monospace">step '+step+'</text>';
  if(hPts.length>1) lhSvg+='<polyline points="'+hPts.join(' ')+'" fill="none" stroke="'+C.accent+'" stroke-width="2"/>';
  var lastPt=hPts[hPts.length-1].split(',');
  lhSvg+='<circle cx="'+lastPt[0]+'" cy="'+lastPt[1]+'" r="3.5" fill="'+C.accent+'"/>';

  var isRun=S.gdRunning;
  var out=sectionTitle('Gradient Descent Training','Watch BCE loss decrease as weights converge to the global optimum');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','BCE Loss Landscape')
    +svgBox(sv,VW,VH)
    +sliderRow('gdLr',S.gdLr,0.05,1.0,0.05,'learning rate α',2)
    +'<div style="display:flex;gap:8px;margin-top:14px;justify-content:center;">'
    +'<button data-action="gdRun" style="padding:8px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+(isRun?hex(C.red,.15):hex(C.accent,.15))+';border:1.5px solid '+(isRun?C.red:C.accent)+';color:'+(isRun?C.red:C.accent)+';cursor:pointer;">'+(isRun?'⏹ Stop':'▶ Run')+'</button>'
    +'<button data-action="gdStep" style="padding:8px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+hex(C.purple,.15)+';border:1.5px solid '+C.purple+';color:'+C.purple+';cursor:pointer;">Step</button>'
    +'<button data-action="gdReset" style="padding:8px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+hex(C.muted,.1)+';border:1.5px solid '+C.dim+';color:'+C.muted+';cursor:pointer;">Reset</button>'
    +'</div>'
  );
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Loss History')
    +'<svg width="100%" viewBox="0 0 '+LW+' '+(LH+20)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+lhSvg+'</svg>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','CURRENT STATE')
    +div('font-size:9px;color:'+C.muted+';line-height:2.1;',
      'Steps: <span style="color:'+C.text+';font-weight:700;">'+step+'</span><br>'
      +'w&#8321;: <span style="color:'+C.accent+';font-weight:700;font-family:monospace;">'+cur.w.toFixed(4)+'</span><br>'
      +'BCE: <span style="color:'+C.yellow+';font-weight:700;font-family:monospace;">'+cur.bce.toFixed(4)+'</span><br>'
      +'Optimal w&#8321;: <span style="color:'+C.green+';font-weight:700;font-family:monospace;">'+_bestW.toFixed(3)+'</span><br>'
      +'Min BCE: <span style="color:'+C.green+';font-weight:700;font-family:monospace;">'+_bestL.toFixed(4)+'</span>'
    ),'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','UPDATE RULE')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;font-family:monospace;">'
    +div('font-size:9px;color:'+C.teal+';font-weight:700;line-height:2.2;','&#8711;L = (1/n) X&#7488;(ŷ &#8722; y)')
    +div('font-size:9px;color:'+C.purple+';font-weight:700;','w := w &#8722; &#945; &#xB7; &#8711;L')
    +'</div>'
    +div('font-size:9px;color:'+C.muted+';margin-top:8px;','&#945; = '+S.gdLr.toFixed(2)+' (learning rate)')
    +div('font-size:8.5px;color:'+C.dim+';margin-top:4px;','Identical form to linear regression — only ŷ = σ(Xw) differs.')
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','LEARNING RATE GUIDE')
    +[
      {lr:'α too small',d:'Slow convergence',c:C.blue},
      {lr:'α just right',d:'Smooth convergence',c:C.green},
      {lr:'α too large',d:'Overshoots / diverges',c:C.red},
    ].map(function(r){
      return '<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:9px;border-bottom:1px solid '+C.border+';">'
        +'<span style="color:'+r.c+';font-weight:700;font-family:monospace;">'+r.lr+'</span>'
        +'<span style="color:'+C.muted+';">'+r.d+'</span></div>';
    }).join(''),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('🎯','Convex = Guaranteed Global Minimum',
    'Because BCE + sigmoid is <span style="color:'+C.green+';font-weight:700;">convex</span>, the loss surface is a smooth bowl with exactly '
    +'<span style="color:'+C.accent+';font-weight:700;">one global minimum</span>. '
    +'Unlike neural networks, logistic regression cannot get trapped in local minima. '
    +'Use <em>Step</em> to advance one iteration, <em>Run</em> to animate, and adjust α to see how the learning rate affects convergence.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 5 — SOFTMAX / MULTI-CLASS
═══════════════════════════════════════════ */
function renderSoftmax(){
  var zs=S.smZ;
  var exps=zs.map(function(z){return Math.exp(Math.max(-20,Math.min(20,z)));});
  var sumExp=exps.reduce(function(a,b){return a+b;},0);
  var probs=exps.map(function(e){return e/sumExp;});
  var maxIdx=probs.indexOf(Math.max.apply(null,probs));
  var classNames=['Class A','Class B','Class C'];
  var classColors=[C.accent,C.blue,C.teal];

  /* bar chart SVG */
  var VW=420,VH=200,PL=80,PR=20,PT=30,PB=30;
  var PW=VW-PL-PR,PH=VH-PT-PB;
  var sv='';
  sv+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,0.25,0.5,0.75,1].forEach(function(v){
    var y=PT+PH*(1-v);
    sv+='<line x1="'+(PL-4)+'" y1="'+y.toFixed(1)+'" x2="'+PL+'" y2="'+y.toFixed(1)+'" stroke="'+C.dim+'" stroke-width="1"/>';
    sv+='<text x="'+(PL-6)+'" y="'+(y+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="7" font-family="monospace">'+(v*100)+'%</text>';
  });
  var bw=PW/3-16;
  classNames.forEach(function(name,i){
    var barH=probs[i]*PH;
    var bx=PL+(i*(PW/3))+8;
    var by=PT+PH-barH;
    var col=classColors[i];
    var winner=i===maxIdx;
    sv+='<rect x="'+bx.toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+barH.toFixed(1)+'" rx="5" fill="'+col+'" opacity="'+(winner?'0.9':'0.45')+'"/>';
    if(winner) sv+='<rect x="'+bx.toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+barH.toFixed(1)+'" rx="5" fill="none" stroke="'+col+'" stroke-width="2"/>';
    sv+='<text x="'+(bx+bw/2).toFixed(1)+'" y="'+(by-6).toFixed(1)+'" text-anchor="middle" fill="'+(winner?col:C.muted)+'" font-size="'+(winner?11:9)+'" font-family="monospace" font-weight="'+(winner?'800':'400')+'">'+(probs[i]*100).toFixed(1)+'%</text>';
    if(winner) sv+='<text x="'+(bx+bw/2).toFixed(1)+'" y="'+(by-20).toFixed(1)+'" text-anchor="middle" font-size="13">&#128081;</text>';
    sv+='<text x="'+(bx+bw/2).toFixed(1)+'" y="'+(PT+PH+18)+'" text-anchor="middle" fill="'+(winner?col:C.muted)+'" font-size="9" font-family="monospace">'+name+(winner?' ✓':'')+'</text>';
  });

  var out=sectionTitle('Multi-Class: Softmax Regression','Generalise logistic regression from 2 classes to k classes');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Adjust Raw Scores (z values)')
    +sliderRow('sm0',zs[0],-3,5,0.1,'z&#8321; (Class A)',1)
    +sliderRow('sm1',zs[1],-3,5,0.1,'z&#8322; (Class B)',1)
    +sliderRow('sm2',zs[2],-3,5,0.1,'z&#8323; (Class C)',1)
    +div('font-size:9px;color:'+C.dim+';margin-top:10px;text-align:center;','← drag to see probabilities update live')
  );
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Softmax Output')
    +'<svg width="100%" viewBox="0 0 '+VW+' '+VH+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv+'</svg>'
    +'<div style="text-align:center;margin-top:8px;font-size:9px;color:'+C.muted+';">Sum = <span style="color:'+C.green+';font-weight:700;">'+probs.reduce(function(a,b){return a+b;},0).toFixed(6)+'</span> (always = 1.000000)</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 210px;display:flex;flex-direction:column;gap:12px;">';
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','SOFTMAX FORMULA')
    +'<div style="background:#08080d;border-radius:8px;padding:10px;text-align:center;font-family:monospace;">'
    +div('font-size:9px;color:'+C.accent+';font-weight:700;line-height:2.2;','P(y=j|x) = e^z&#11388; / &#8721;&#8342; e^z&#8342;')
    +'</div>','max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','STEP-BY-STEP')
    +div('font-size:8.5px;font-weight:700;color:'+C.muted+';margin-bottom:4px;','1. RAW SCORES z:')
    +classNames.map(function(n,i){
      return '<div style="display:flex;justify-content:space-between;font-size:9px;font-family:monospace;padding:1px 0;">'
        +'<span style="color:'+classColors[i]+';">z'+(i+1)+'</span>'
        +'<span style="color:'+C.text+';">= '+zs[i].toFixed(1)+'</span></div>';
    }).join('')
    +div('font-size:8.5px;font-weight:700;color:'+C.muted+';margin:8px 0 4px;','2. EXPONENTIATE:')
    +classNames.map(function(n,i){
      return '<div style="display:flex;justify-content:space-between;font-size:9px;font-family:monospace;padding:1px 0;">'
        +'<span style="color:'+classColors[i]+';">e^'+zs[i].toFixed(1)+'</span>'
        +'<span style="color:'+C.text+';">= '+exps[i].toFixed(3)+'</span></div>';
    }).join('')
    +div('font-size:8.5px;font-weight:700;color:'+C.muted+';margin:8px 0 4px;','3. SUM: <span style="color:'+C.yellow+';">'+sumExp.toFixed(3)+'</span>')
    +div('font-size:8.5px;font-weight:700;color:'+C.muted+';margin-bottom:4px;','4. DIVIDE:')
    +classNames.map(function(n,i){
      var w=i===maxIdx;
      return '<div style="display:flex;justify-content:space-between;font-size:9px;font-family:monospace;padding:1px 0;">'
        +'<span style="color:'+classColors[i]+';">P('+n+')</span>'
        +'<span style="color:'+(w?classColors[i]:C.text)+';font-weight:'+(w?'800':'400')+';">'+(probs[i]*100).toFixed(1)+'%'+(w?' ✓':'')+'</span></div>';
    }).join('')
    ,'max-width:none;margin:0 0 12px;'
  );
  out+=card(
    div('font-size:9px;color:'+C.muted+';margin-bottom:8px;','OvR vs SOFTMAX')
    +div('font-size:9px;color:'+C.muted+';line-height:1.9;',
      '<span style="color:'+C.purple+';font-weight:700;">One-vs-Rest:</span><br>'
      +'k independent sigmoids.<br>'
      +'Probs <span style="color:'+C.red+';">don\'t</span> sum to 1.<br><br>'
      +'<span style="color:'+C.accent+';font-weight:700;">Softmax:</span><br>'
      +'One joint model with k weight vectors.<br>'
      +'Probs <span style="color:'+C.green+';">always</span> sum to 1.'
    ),'max-width:none;margin:0;'
  );
  out+='</div></div>';
  out+=insight('🧠','Softmax = Logistic Regression at Scale',
    'When k=2, softmax gives <span style="color:'+C.accent+';font-weight:700;">exactly the same result as the sigmoid</span>. '
    +'For k &gt; 2, softmax is the canonical multi-class extension: one weight vector per class, '
    +'trained with <span style="color:'+C.teal+';font-weight:700;">categorical cross-entropy</span>. '
    +'The output layer of every multi-class neural network <em>is</em> softmax regression.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════ */
var TABS=[
  '&#963; Sigmoid',
  '&#128208; Decision Boundary',
  '&#128201; BCE Loss',
  '&#8711; Gradient Descent',
  '&#8805;3 Softmax'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.purple+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Logistic Regression</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Interactive visual walkthrough &#8212; sigmoid &#xB7; decision boundary &#xB7; BCE &#xB7; gradient descent &#xB7; softmax')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  if(S.tab===0) html+=renderSigmoid();
  else if(S.tab===1) html+=renderDecisionBoundary();
  else if(S.tab===2) html+=renderLoss();
  else if(S.tab===3) html+=renderGradient();
  else if(S.tab===4) html+=renderSoftmax();
  html+='</div>';
  return html;
}

/* ─── GD STEP ─── */
function doGdStep(){
  if(S.gdHist.length>=65){
    S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;render();return;
  }
  var last=S.gdHist[S.gdHist.length-1];
  var g=grad1d(last.w);
  var nw=last.w-S.gdLr*g;
  S.gdHist=S.gdHist.concat([{w:nw,bce:bce1d(nw)}]);
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
          S.tab=idx;render();
        }
        else if(action==='gdRun'){
          S.gdRunning=!S.gdRunning;
          if(S.gdRunning) S.gdTimer=setInterval(doGdStep,260);
          else{clearInterval(S.gdTimer);S.gdTimer=null;}
          render();
        }
        else if(action==='gdStep'){if(!S.gdRunning) doGdStep();}
        else if(action==='gdReset'){
          S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;
          S.gdHist=[{w:3.5,bce:bce1d(3.5)}];render();
        }
      });
    }
    else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='sigZ') S.sigZ=val;
        else if(action==='dbW1') S.dbW1=val;
        else if(action==='dbW2') S.dbW2=val;
        else if(action==='dbB')  S.dbB=val;
        else if(action==='bceYhat') S.bceYhat=val;
        else if(action==='gdLr'){
          S.gdRunning=false;clearInterval(S.gdTimer);S.gdTimer=null;
          S.gdHist=[{w:3.5,bce:bce1d(3.5)}];
          S.gdLr=val;
        }
        else if(action==='sm0') S.smZ=[val,S.smZ[1],S.smZ[2]];
        else if(action==='sm1') S.smZ=[S.smZ[0],val,S.smZ[2]];
        else if(action==='sm2') S.smZ=[S.smZ[0],S.smZ[1],val];
        render();
      });
    }
  });
}

render();
</script>
</body>
</html>"""

LOR_VISUAL_HEIGHT = 1100