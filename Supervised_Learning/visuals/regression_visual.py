"""
Self-contained HTML visual for Regression (Overview + ASIC Method Map).
4 interactive tabs: What is Regression?, Method Map (ASIC), Family Breakdown, When to Use.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(REG_VISUAL_HTML, height=REG_VISUAL_HEIGHT, scrolling=True)
"""

REG_VISUAL_HTML = r"""<!DOCTYPE html>
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
  orange:"#fb923c",pink:"#f472b6"};

/* ─── STATE ─── */
var S={
  tab:0,
  selectedFamily:-1,
  hoveredNode:-1,
  noise:0.8,
  degree:1
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

/* ─── SVG SCAFFOLD ─── */
var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
var PW=VW-PL-PR,PH=VH-PT-PB;
function sx(x,xmin,xmax){return PL+((x-xmin)/(xmax-xmin))*PW;}
function sy(y,ymin,ymax){return PT+PH-((y-ymin)/(ymax-ymin))*PH;}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||VW)+' '+(h||VH)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}
function plotGrid(xlabel,ylabel,xmin,xmax,ymin,ymax,xticks,yticks){
  var o='';
  xticks.forEach(function(v){
    o+='<line x1="'+sx(v,xmin,xmax).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xmin,xmax).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+sx(v,xmin,xmax).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  yticks.forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+sy(v,ymin,ymax).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ymin,ymax).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+(PL-6)+'" y="'+(sy(v,ymin,ymax)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';
  });
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xlabel+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+ylabel+'</text>';
  return o;
}

/* ─── DATA GEN ─── */
function genData(n,fn,noise){
  var pts=[];
  for(var i=0;i<n;i++){
    var x=i/(n-1)*10;
    var y=fn(x)+(Math.random()-0.5)*2*noise;
    pts.push({x:x,y:y});
  }
  return pts;
}

/* ═══════════════════════════════════════════
   TAB 1 — WHAT IS REGRESSION
═══════════════════════════════════════════ */
function renderWhat(){
  var d=S.degree;
  var noise=S.noise;

  /* generate sample data: underlying truth is sin-ish curve */
  var seed=42;
  function seededRand(i){var x=Math.sin(seed+i*127.1)*43758.5453123;return x-Math.floor(x);}
  var rawPts=[];
  for(var i=0;i<18;i++){
    var x=0.5+i*0.5;
    var yTrue=1.5*x+2+1.5*Math.sin(x*0.8);
    var y=yTrue+(seededRand(i)-0.5)*2*noise;
    rawPts.push({x:x,y:y});
  }

  /* fit polynomial of degree d via least squares (up to degree 3) */
  function polyPredict(x,coeffs){
    var y=0;for(var i=0;i<coeffs.length;i++) y+=coeffs[i]*Math.pow(x,i);return y;
  }
  /* simple closed-form for degree 1 */
  function fitLinear(pts){
    var n=pts.length,sx=0,sy=0,sxy=0,sx2=0;
    pts.forEach(function(p){sx+=p.x;sy+=p.y;sxy+=p.x*p.y;sx2+=p.x*p.x;});
    var m=(n*sxy-sx*sy)/(n*sx2-sx*sx);
    var b=(sy-m*sx)/n;
    return [b,m];
  }
  function fitPoly(pts,deg){
    /* Vandermonde normal equations solved with simple Gaussian elimination */
    var n=pts.length,k=deg+1;
    var A=[];
    for(var r=0;r<k;r++){
      A[r]=[];
      for(var c2=0;c2<k+1;c2++) A[r][c2]=0;
    }
    pts.forEach(function(p){
      for(var r=0;r<k;r++){
        for(var c2=0;c2<k;c2++) A[r][c2]+=Math.pow(p.x,r+c2);
        A[r][k]+=p.y*Math.pow(p.x,r);
      }
    });
    /* forward elimination */
    for(var col=0;col<k;col++){
      var pivot=col;
      for(var row=col+1;row<k;row++) if(Math.abs(A[row][col])>Math.abs(A[pivot][col])) pivot=row;
      var tmp=A[col];A[col]=A[pivot];A[pivot]=tmp;
      for(var row=col+1;row<k;row++){
        var f=A[row][col]/A[col][col];
        for(var c2=col;c2<=k;c2++) A[row][c2]-=f*A[col][c2];
      }
    }
    /* back substitution */
    var x2=new Array(k);
    for(var row=k-1;row>=0;row--){
      x2[row]=A[row][k];
      for(var c2=row+1;c2<k;c2++) x2[row]-=A[row][c2]*x2[c2];
      x2[row]/=A[row][row];
    }
    return x2;
  }

  var coeffs=d===1?fitLinear(rawPts):fitPoly(rawPts,d);
  var xmin=0,xmax=10,ymin=-1,ymax=18;
  var xt=[0,2,4,6,8,10],yt=[0,4,8,12,16];
  var sv=plotGrid('Input x','Output y',xmin,xmax,ymin,ymax,xt,yt);

  /* draw fit curve */
  var steps=80;
  var linePts=[];
  for(var i2=0;i2<=steps;i2++){
    var xv=xmin+i2*(xmax-xmin)/steps;
    var yv=polyPredict(xv,coeffs);
    linePts.push([sx(xv,xmin,xmax),sy(yv,ymin,ymax)]);
  }
  var pathD='M'+linePts[0][0].toFixed(1)+','+Math.max(PT,Math.min(PT+PH,linePts[0][1])).toFixed(1);
  for(var i2=1;i2<linePts.length;i2++){
    var cy2=Math.max(PT,Math.min(PT+PH,linePts[i2][1]));
    pathD+=' L'+linePts[i2][0].toFixed(1)+','+cy2.toFixed(1);
  }
  var lineColor=d===1?C.accent:d===2?C.purple:C.orange;
  sv+='<path d="'+pathD+'" fill="none" stroke="'+lineColor+'" stroke-width="2.5"/>';

  /* residuals */
  rawPts.forEach(function(p){
    var yh=polyPredict(p.x,coeffs);
    sv+='<line x1="'+sx(p.x,xmin,xmax).toFixed(1)+'" y1="'+sy(p.y,ymin,ymax).toFixed(1)+'"'
      +' x2="'+sx(p.x,xmin,xmax).toFixed(1)+'" y2="'+Math.max(PT,Math.min(PT+PH,sy(yh,ymin,ymax))).toFixed(1)+'"'
      +' stroke="'+C.red+'" stroke-width="1" stroke-dasharray="3,2" opacity="0.4"/>';
  });

  /* points */
  rawPts.forEach(function(p){
    sv+='<circle cx="'+sx(p.x,xmin,xmax).toFixed(1)+'" cy="'+sy(p.y,ymin,ymax).toFixed(1)+'"'
      +' r="3.5" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  });

  /* label */
  var degLabel=d===1?'Linear':d===2?'Quadratic':'Cubic';
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="130" height="20" rx="4" fill="#0a0a0f" opacity="0.88"/>';
  sv+='<text x="'+(PL+10)+'" y="'+(PT+15)+'" fill="'+lineColor+'" font-size="9.5" font-family="monospace" font-weight="700">'+degLabel+' fit  (degree='+d+')</text>';

  var out=sectionTitle('What is Regression?','Predicting a continuous numerical output from one or more inputs');

  /* concept cards row */
  out+='<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  [
    {icon:'🎯',title:'Goal',body:'Learn a function  f(x) → y  that maps inputs to a continuous number',c:C.accent},
    {icon:'📉',title:'Loss',body:'Minimise prediction error — typically Mean Squared Error or MAE',c:C.yellow},
    {icon:'📐',title:'Output',body:'A real number: price, temperature, probability score, distance…',c:C.blue},
  ].forEach(function(item){
    out+='<div style="flex:1 1 200px;background:#08080d;border:1px solid '+hex(item.c,.25)+';border-radius:8px;padding:14px;">'
      +'<div style="font-size:16px;margin-bottom:6px;">'+item.icon+'</div>'
      +'<div style="font-size:10px;font-weight:700;color:'+item.c+';margin-bottom:6px;">'+item.title+'</div>'
      +'<div style="font-size:9.5px;color:'+C.muted+';line-height:1.7;">'+item.body+'</div>'
      +'</div>';
  });
  out+='</div>';

  /* interactive plot */
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Adjustable Fit — change complexity & noise')
    +svgBox(sv)
    +sliderRow('degree',d,1,3,1,'poly degree',0)
    +sliderRow('noise',noise,0.1,3,0.1,'noise σ',1)
  );

  /* regression vs classification */
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Regression vs Classification')
    +'<div style="display:flex;gap:8px;flex-wrap:wrap;">'
    +'<div style="flex:1 1 45%;padding:12px;background:#08080d;border-radius:8px;border:1px solid '+hex(C.accent,.25)+';min-width:180px;">'
    +'<div style="font-size:10px;font-weight:700;color:'+C.accent+';margin-bottom:8px;">REGRESSION  📈</div>'
    +['Output: continuous number','Ask: "How much?" / "How many?"','Loss: MSE, MAE, Huber','e.g. House price → $342,000','e.g. Temperature → 22.4°C'].map(function(t){
      return '<div style="font-size:9px;color:'+C.muted+';padding:3px 0;border-bottom:1px solid '+C.border+';">'+t+'</div>';
    }).join('')+'</div>'
    +'<div style="flex:1 1 45%;padding:12px;background:#08080d;border-radius:8px;border:1px solid '+hex(C.purple,.25)+';min-width:180px;">'
    +'<div style="font-size:10px;font-weight:700;color:'+C.purple+';margin-bottom:8px;">CLASSIFICATION  🏷️</div>'
    +['Output: discrete category','Ask: "Which class?"','Loss: Cross-entropy, Hinge','e.g. Email → Spam / Not Spam','e.g. Image → Cat / Dog / Bird'].map(function(t){
      return '<div style="font-size:9px;color:'+C.muted+';padding:3px 0;border-bottom:1px solid '+C.border+';">'+t+'</div>';
    }).join('')+'</div>'
    +'</div>'
  );

  out+=insight('💡','The Core Equation',
    'All regression reduces to learning <span style="color:'+C.accent+';font-weight:700;">ŷ = f(x; θ)</span> where θ are learnable parameters. '
    +'The choice of <span style="color:'+C.blue+';font-weight:700;">f</span> (linear, tree, neural net…) and '
    +'<span style="color:'+C.yellow+';font-weight:700;">loss function</span> defines which regression method you are using.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 2 — ASIC METHOD MAP
═══════════════════════════════════════════ */
function renderASIC(){
  var W=900,H=820;

  /* ─── node layout ─── */
  var families=[
    {
      id:0,label:'LINEAR MODELS',color:C.accent,x:30,y:100,w:155,
      methods:['Simple Linear','Multiple Linear','Polynomial','Ridge  (L2)','Lasso  (L1)','ElasticNet','Bayesian Linear']
    },
    {
      id:1,label:'REGULARISED',color:C.blue,x:215,y:100,w:155,
      methods:['Ridge','Lasso','ElasticNet','ARD Regression','Lars','BayesianRidge','Huber Regressor']
    },
    {
      id:2,label:'TREE-BASED',color:C.green,x:400,y:100,w:155,
      methods:['Decision Tree','Random Forest','Extra Trees','Gradient Boost','XGBoost','LightGBM','CatBoost']
    },
    {
      id:3,label:'KERNEL / SVM',color:C.purple,x:585,y:100,w:155,
      methods:['Linear SVR','Kernel SVR','RBF Kernel','Poly Kernel','Nu-SVR','Gaussian Proc.','Kriging']
    },
    {
      id:4,label:'NEURAL NETS',color:C.orange,x:770,y:100,w:120,
      methods:['MLP Regressor','CNN Regressor','LSTM / GRU','Transformer','Attention Net','TabNet','ResNet Tab']
    },
    {
      id:5,label:'NEIGHBOURS',color:C.yellow,x:30,y:490,w:155,
      methods:['KNN Regressor','Weighted KNN','Radius Neighbor','Kernel Smooth.','LOWESS','NW Estimator']
    },
    {
      id:6,label:'DIMENSION RED.',color:C.pink,x:215,y:490,w:155,
      methods:['PCR','Partial Least Sq','PLSR','OPLS','CCA Regressor','Factor Regress.']
    },
    {
      id:7,label:'GENERALIZED',color:C.red,x:400,y:490,w:155,
      methods:['Poisson Regress.','Neg. Binomial','Gamma Regress.','Tweedie','Beta Regress.','Quantile Regr.','Tobit Model']
    },
    {
      id:8,label:'TIME SERIES',color:'#67e8f9',x:585,y:490,w:155,
      methods:['ARIMA','SARIMA','ARIMAX','VAR','Prophet','LSTM-TS','Theta Model']
    },
    {
      id:9,label:'ENSEMBLE',color:'#d4a574',x:770,y:490,w:120,
      methods:['Voting Reg.','Stacking','Blending','Bagging Reg.','BMA','Super Learner']
    }
  ];

  var sel=S.selectedFamily;

  var svg='<defs>';
  /* glow filters */
  families.forEach(function(f){
    var id='glow'+f.id;
    svg+='<filter id="'+id+'" x="-30%" y="-30%" width="160%" height="160%">'
      +'<feGaussianBlur stdDeviation="4" result="blur"/>'
      +'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
      +'</filter>';
  });
  svg+='</defs>';

  /* ── background chip body ── */
  svg+='<rect x="8" y="8" width="884" height="804" rx="18" fill="#0c0c14" stroke="'+C.border+'" stroke-width="1.5"/>';

  /* chip label */
  svg+='<text x="450" y="36" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" letter-spacing="4">REGRESSION ENGINE  ·  v1.0  ·  METHOD MAP</text>';

  /* central bus line */
  svg+='<rect x="18" y="56" width="864" height="2" rx="1" fill="'+C.border+'"/>';
  svg+='<rect x="18" y="442" width="864" height="2" rx="1" fill="'+C.border+'"/>';
  svg+='<text x="450" y="454" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace" letter-spacing="2">GENERAL  PURPOSE  REGRESSION  BUS</text>';
  svg+='<text x="450" y="68" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace" letter-spacing="2">PARAMETRIC  REGRESSION  BUS</text>';

  /* vertical bus connectors between rows */
  [110,295,480,665].forEach(function(bx){
    svg+='<line x1="'+bx+'" y1="58" x2="'+bx+'" y2="442" stroke="'+C.border+'" stroke-width="0.8" stroke-dasharray="4,4"/>';
  });

  /* row labels */
  svg+='<text x="20" y="96" fill="'+C.dim+'" font-size="8" font-family="monospace" letter-spacing="1">FAMILY A  ▸  PARAMETRIC</text>';
  svg+='<text x="20" y="486" fill="'+C.dim+'" font-size="8" font-family="monospace" letter-spacing="1">FAMILY B  ▸  NON-PARAMETRIC &amp; SPECIALISED</text>';

  /* output bus at bottom */
  svg+='<rect x="200" y="770" width="500" height="28" rx="6" fill="#0e1520" stroke="'+hex(C.accent,.3)+'" stroke-width="1"/>';
  svg+='<text x="450" y="789" text-anchor="middle" fill="'+C.accent+'" font-size="9" font-family="monospace" font-weight="700">OUTPUT  →  ŷ  (continuous real number)</text>';

  /* ── draw families ── */
  families.forEach(function(f){
    var isTop=f.y<400;
    var boxH=isTop?320:280;
    var rowH=16;
    var highlight=sel===f.id;
    var alpha=sel>=0&&!highlight?0.35:1;
    var fc=f.color;

    /* connector line to bus */
    var busY=isTop?58:442;
    var midX=f.x+f.w/2;
    svg+='<line x1="'+midX+'" y1="'+f.y+'" x2="'+midX+'" y2="'+busY+'"'
      +' stroke="'+fc+'" stroke-width="'+(highlight?1.5:0.8)+'" opacity="'+(alpha*0.6)+'" stroke-dasharray="'+(isTop?'':'3,3')+'"/>';

    /* connector line down to output bus */
    if(highlight){
      svg+='<line x1="'+midX+'" y1="'+(f.y+boxH)+'" x2="'+midX+'" y2="770"'
        +' stroke="'+fc+'" stroke-width="1.2" opacity="0.4" stroke-dasharray="4,4"/>';
    }

    /* family box */
    svg+='<rect x="'+f.x+'" y="'+f.y+'" width="'+f.w+'" height="'+boxH+'" rx="10"'
      +' fill="'+hex(fc,highlight?0.08:0.04)+'" stroke="'+fc+'" stroke-width="'+(highlight?2:1)+'" opacity="'+alpha+'"'
      +(highlight?' filter="url(#glow'+f.id+')"':'')+'/>';

    /* family header */
    svg+='<rect x="'+f.x+'" y="'+f.y+'" width="'+f.w+'" height="26" rx="10" fill="'+hex(fc,.18)+'"/>';
    svg+='<rect x="'+f.x+'" y="'+(f.y+16)+'" width="'+f.w+'" height="10" fill="'+hex(fc,.18)+'"/>';
    svg+='<text x="'+(f.x+f.w/2)+'" y="'+(f.y+17)+'" text-anchor="middle" fill="'+fc+'" font-size="8.5" font-family="monospace" font-weight="700" opacity="'+alpha+'">'+f.label+'</text>';

    /* methods */
    f.methods.forEach(function(m,mi){
      var my=f.y+32+mi*rowH;
      var mhighlight=highlight;
      svg+='<rect x="'+(f.x+6)+'" y="'+my+'" width="'+(f.w-12)+'" height="13" rx="3"'
        +' fill="'+hex(fc,mhighlight?0.12:0.0)+'" opacity="'+alpha+'"/>';
      svg+='<circle cx="'+(f.x+14)+'" cy="'+(my+6.5)+'" r="2.5" fill="'+fc+'" opacity="'+(alpha*0.7)+'"/>';
      svg+='<text x="'+(f.x+22)+'" y="'+(my+9)+'" fill="'+(mhighlight?C.text:C.muted)+'" font-size="8.5" font-family="monospace" opacity="'+alpha+'">'+m+'</text>';
    });

    /* clickable overlay */
    svg+='<rect x="'+f.x+'" y="'+f.y+'" width="'+f.w+'" height="'+boxH+'" rx="10" fill="transparent"'
      +' data-action="selFamily" data-idx="'+f.id+'" style="cursor:pointer;"/>';
  });

  /* centre labels */
  svg+='<text x="450" y="620" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">CLICK ANY BLOCK TO HIGHLIGHT</text>';

  /* reset hint */
  if(sel>=0){
    svg+='<rect x="370" y="630" width="160" height="20" rx="4" fill="'+hex(C.accent,.1)+'" stroke="'+hex(C.accent,.3)+'" stroke-width="1" data-action="selFamily" data-idx="-1" style="cursor:pointer;"/>';
    svg+='<text x="450" y="644" text-anchor="middle" fill="'+C.accent+'" font-size="9" font-family="monospace" data-action="selFamily" data-idx="-1" style="cursor:pointer;">RESET SELECTION</text>';
  }

  var out=sectionTitle('Regression Method Map','ASIC-style architecture diagram — all regression families and their algorithms');

  /* detail panel when a family is selected */
  if(sel>=0){
    var sf=families[sel];
    out+='<div style="max-width:750px;margin:0 auto 14px;padding:14px 18px;background:'+hex(sf.color,.06)+';border-radius:10px;border:1px solid '+hex(sf.color,.3)+'">';
    out+='<div style="font-size:10px;font-weight:700;color:'+sf.color+';margin-bottom:8px;">▸ '+sf.label+'</div>';
    out+='<div style="display:flex;flex-wrap:wrap;gap:6px;">';
    sf.methods.forEach(function(m){
      out+='<span style="font-size:9px;padding:3px 10px;border-radius:4px;background:'+hex(sf.color,.12)+';border:1px solid '+hex(sf.color,.25)+';color:'+sf.color+';">'+m+'</span>';
    });
    out+='</div></div>';
  }

  out+='<div style="max-width:900px;margin:0 auto;overflow-x:auto;">';
  out+=svgBox(svg,W,H);
  out+='</div>';
  out+=insight('🗺️','How to Read This Diagram',
    '<span style="color:'+C.accent+';font-weight:700;">Top row (Family A)</span> — Parametric methods: assume a fixed functional form, tend to be faster and more interpretable. '
    +'<span style="color:C.yellow;font-weight:700;"><br><br><span style="color:'+C.yellow+';font-weight:700;">Bottom row (Family B)</span> — Non-parametric & specialised: make fewer assumptions about data shape. Click any block to highlight.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   TAB 3 — FAMILY BREAKDOWN
═══════════════════════════════════════════ */
function renderFamilies(){
  var families=[
    {
      label:'Linear Models',color:C.accent,icon:'📐',
      when:'Relationship between x and y is roughly linear',
      pros:['Highly interpretable (coefficients = feature importance)','Fast to train (closed-form solution)','Works well with small datasets'],
      cons:['Cannot capture non-linear patterns','Sensitive to outliers (especially OLS)','Assumes features are independent'],
      algos:'Simple, Multiple, Polynomial, Ridge, Lasso, ElasticNet'
    },
    {
      label:'Regularised Models',color:C.blue,icon:'⚖️',
      when:'Linear data but with many features or multicollinearity',
      pros:['Prevents overfitting via penalty term','Lasso performs automatic feature selection','ElasticNet balances both L1 and L2'],
      cons:['Hyperparameter tuning required (λ)','Lasso unstable with correlated features','Still assumes linearity'],
      algos:'Ridge (L2), Lasso (L1), ElasticNet, Bayesian Ridge'
    },
    {
      label:'Tree-Based',color:C.green,icon:'🌲',
      when:'Complex non-linear relationships, tabular data',
      pros:['Handles non-linearity natively','Robust to outliers and missing values','Feature importance built-in'],
      cons:['Single trees overfit easily','Less interpretable for deep trees','Boosting can be slow to train'],
      algos:'Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost'
    },
    {
      label:'Kernel / SVM',color:C.purple,icon:'🔮',
      when:'Small-medium datasets with complex boundaries',
      pros:['Effective in high-dimensional spaces','Kernel trick enables non-linear fitting','Robust to outliers (margin-based)'],
      cons:['Slow on large datasets O(n²-n³)','Kernel choice is non-trivial','Memory intensive'],
      algos:'Linear SVR, RBF SVR, Polynomial SVR, Gaussian Processes'
    },
    {
      label:'Neural Networks',color:C.orange,icon:'🧠',
      when:'Large datasets, image/text/sequence inputs',
      pros:['Universal function approximators','Handles unstructured data natively','State-of-art performance at scale'],
      cons:['Requires large amounts of data','Black-box — low interpretability','Compute intensive, many hyperparams'],
      algos:'MLP, CNN, LSTM, Transformer, TabNet, ResNet-Tabular'
    },
    {
      label:'Generalised Linear',color:C.red,icon:'📊',
      when:'Count data, rates, bounded outputs (0–1), skewed targets',
      pros:['Handles non-Gaussian targets','Link function adapts to output distribution','Statistically principled'],
      cons:['Must choose correct distribution family','Less flexible than trees/NNs','Assumes specific error structure'],
      algos:'Poisson, Negative Binomial, Gamma, Tweedie, Quantile Regr.'
    },
  ];

  var out=sectionTitle('Family Breakdown','Key characteristics, tradeoffs, and best-fit scenarios for each family');

  families.forEach(function(f){
    out+=card(
      '<div style="display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;">'
      +'<div style="width:130px;flex-shrink:0;">'
      +'<div style="font-size:22px;margin-bottom:6px;">'+f.icon+'</div>'
      +'<div style="font-size:10px;font-weight:700;color:'+f.color+';margin-bottom:8px;">'+f.label+'</div>'
      +'<div style="font-size:8.5px;padding:5px 8px;background:'+hex(f.color,.1)+';border-radius:4px;border:1px solid '+hex(f.color,.25)+';color:'+f.color+';line-height:1.6;">USE WHEN<br>'+f.when+'</div>'
      +'</div>'
      +'<div style="flex:1;min-width:200px;">'
      +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
      +'<div style="flex:1;min-width:140px;">'
      +'<div style="font-size:9px;font-weight:700;color:'+C.green+';margin-bottom:5px;">✓ STRENGTHS</div>'
      +f.pros.map(function(p){return '<div style="font-size:9px;color:'+C.muted+';padding:2px 0;border-bottom:1px solid #16161e;">'+p+'</div>';}).join('')
      +'</div>'
      +'<div style="flex:1;min-width:140px;">'
      +'<div style="font-size:9px;font-weight:700;color:'+C.red+';margin-bottom:5px;">✗ LIMITATIONS</div>'
      +f.cons.map(function(p){return '<div style="font-size:9px;color:'+C.muted+';padding:2px 0;border-bottom:1px solid #16161e;">'+p+'</div>';}).join('')
      +'</div>'
      +'</div>'
      +'<div style="margin-top:10px;padding:6px 10px;background:#08080d;border-radius:6px;border-left:2px solid '+f.color+';">'
      +'<span style="font-size:8px;color:'+C.dim+';">ALGORITHMS:  </span>'
      +'<span style="font-size:8.5px;color:'+C.muted+';">'+f.algos+'</span>'
      +'</div>'
      +'</div>'
      +'</div>',
      'max-width:none;'
    );
  });
  return out;
}

/* ═══════════════════════════════════════════
   TAB 4 — WHEN TO USE
═══════════════════════════════════════════ */
function renderWhen(){
  var out=sectionTitle('When to Use Which','A decision guide for picking the right regression method');

  /* Decision Flow */
  var W=720,H=620;
  var dfc=C.accent;
  var sv='';

  function node(x,y,w,h,text,color,shape){
    var c=color||C.accent;
    if(shape==='diamond'){
      var hw=w/2,hh=h/2;
      sv+='<polygon points="'+x+','+(y-hh)+' '+(x+hw)+','+y+' '+x+','+(y+hh)+' '+(x-hw)+','+y+'"'
        +' fill="'+hex(c,.12)+'" stroke="'+c+'" stroke-width="1.5"/>';
      text.split('|').forEach(function(line,li){
        sv+='<text x="'+x+'" y="'+(y-5+(li*12))+'" text-anchor="middle" fill="'+c+'" font-size="8.5" font-family="monospace">'+line+'</text>';
      });
    } else {
      sv+='<rect x="'+(x-w/2)+'" y="'+(y-h/2)+'" width="'+w+'" height="'+h+'" rx="6"'
        +' fill="'+hex(c,.12)+'" stroke="'+c+'" stroke-width="1.5"/>';
      text.split('|').forEach(function(line,li){
        var totalLines=text.split('|').length;
        sv+='<text x="'+x+'" y="'+(y-((totalLines-1)*6)+(li*13))+'" text-anchor="middle" fill="'+c+'" font-size="8.5" font-family="monospace">'+line+'</text>';
      });
    }
  }
  function arrow(x1,y1,x2,y2,label,lc){
    sv+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+C.dim+'" stroke-width="1.2"/>';
    sv+='<polygon points="'+x2+','+(y2)+' '+(x2-4)+','+(y2-6)+' '+(x2+4)+','+(y2-6)+'" fill="'+C.dim+'"/>';
    if(label){
      var mx=(x1+x2)/2+6,my=(y1+y2)/2;
      sv+='<text x="'+mx+'" y="'+my+'" fill="'+(lc||C.yellow)+'" font-size="8" font-family="monospace">'+label+'</text>';
    }
  }
  function harrow(x1,y1,x2,y2,label,lc){
    sv+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+C.dim+'" stroke-width="1.2"/>';
    sv+='<polygon points="'+x2+','+y2+' '+(x2-6)+','+(y2-4)+' '+(x2-6)+','+(y2+4)+'" fill="'+C.dim+'"/>';
    if(label){
      var mx=(x1+x2)/2,my=y1-8;
      sv+='<text x="'+mx+'" y="'+my+'" text-anchor="middle" fill="'+(lc||C.yellow)+'" font-size="8" font-family="monospace">'+label+'</text>';
    }
  }

  /* Start */
  node(360,38,160,26,'START: Regression Problem',C.accent);
  arrow(360,51,360,90,'');

  /* Q1: linear? */
  node(360,110,120,40,'Linear|relationship?',C.yellow,'diamond');
  arrow(360,130,360,168,'YES',C.green);
  harrow(360,110,540,110,'NO',C.red);

  /* Linear branch */
  node(360,185,130,26,'Many features?',C.yellow,'diamond');
  arrow(360,198,360,238,'YES',C.green);
  harrow(360,185,180,185,'NO',C.red);

  node(180,185,130,26,'Use: Linear Reg',C.accent);
  node(360,255,150,26,'Correlated feats?',C.yellow,'diamond');
  arrow(360,268,360,308,'YES',C.green);
  harrow(360,255,565,255,'NO',C.red);

  node(565,255,120,26,'Use: Lasso / PCR',C.blue);
  node(360,325,140,26,'Use: ElasticNet',C.blue);

  /* Non-linear branch */
  node(540,130,130,26,'Large dataset?',C.yellow,'diamond');
  arrow(540,143,540,180,'YES',C.green);
  harrow(540,130,680,130,'NO',C.red);

  node(680,130,110,26,'Small data?',C.yellow,'diamond');
  arrow(680,143,680,180,'YES',C.green);
  node(680,196,120,26,'Use: Kernel SVR|or KNN',C.purple);

  node(540,196,130,26,'Structured/Tabular?',C.yellow,'diamond');
  arrow(540,209,540,248,'YES',C.green);
  harrow(540,196,680,196,'NO',C.red);

  node(540,265,140,26,'Use: XGBoost / LightGBM',C.green);

  node(680,212,120,26,'Image/Text/Seq?',C.yellow,'diamond');
  arrow(680,225,680,264,'YES',C.green);
  node(680,280,120,26,'Use: Neural Nets|(CNN/LSTM)',C.orange);

  /* Q: special outputs */
  node(360,400,150,26,'Special output type?',C.yellow,'diamond');
  arrow(360,338,360,400,'');
  /* connect from start through linear */
  sv+='<line x1="200" y1="400" x2="360" y2="400" stroke="'+C.dim+'" stroke-width="0.8" stroke-dasharray="3,3"/>';

  harrow(360,400,160,400,'Counts/Rates',C.red);
  node(160,400,120,26,'Use: Poisson|or Neg. Binomial',C.red);

  harrow(360,400,560,400,'Temporal/TS',C.red);
  node(560,400,120,26,'Use: ARIMA|Prophet  LSTM-TS',C.cyan||'#67e8f9');

  arrow(360,413,360,456,'Quantiles');
  node(360,472,140,26,'Use: Quantile Regression',C.pink);

  /* always ensemble note */
  node(360,550,250,36,'Always try: Ensemble (stacking / blending)|when maximising predictive accuracy',C.orange);

  sv+=svgBox(sv,W,H);

  out+='<div style="max-width:750px;margin:0 auto 14px;overflow-x:auto;">';
  out+=svgBox(sv,W,H);
  out+='</div>';

  /* Cheat sheet table */
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Quick Reference Cheat Sheet')
    +'<table style="width:100%;border-collapse:collapse;font-size:9px;">'
    +'<tr style="border-bottom:1px solid '+C.border+';">'
    +'<th style="text-align:left;padding:6px 8px;color:'+C.muted+';font-weight:700;">SCENARIO</th>'
    +'<th style="text-align:left;padding:6px 8px;color:'+C.muted+';font-weight:700;">FIRST TRY</th>'
    +'<th style="text-align:left;padding:6px 8px;color:'+C.muted+';font-weight:700;">IF MORE POWER NEEDED</th>'
    +'</tr>'
    +[
      ['Linear data, few features',            'OLS Linear Regression',       'Ridge / Bayesian Linear'],
      ['Linear data, many features',           'Lasso Regression',            'ElasticNet / PCR'],
      ['Non-linear, tabular, medium data',     'Random Forest Regressor',     'XGBoost / LightGBM'],
      ['Non-linear, tabular, large data',      'LightGBM',                    'Stacking Ensemble'],
      ['Images as input',                      'CNN Regressor',               'Fine-tuned ResNet/ViT'],
      ['Text as input',                        'TF-IDF + Ridge',              'Fine-tuned Transformer'],
      ['Time series forecasting',              'ARIMA / Prophet',             'LSTM / Temporal Fusion'],
      ['Count / rate targets',                 'Poisson Regression',          'Negative Binomial / Tweedie'],
      ['Need prediction intervals',            'Quantile Regression',         'Conformal Prediction'],
      ['Very small dataset (<200 rows)',        'Ridge / Bayesian Ridge',      'Gaussian Process Regression'],
      ['Need interpretability',                'Linear Regression',           'Decision Tree / SHAP on RF'],
      ['Maximum accuracy (competition)',       'XGBoost + LightGBM stack',    'Neural AutoML ensemble'],
    ].map(function(row,ri){
      var bg=ri%2===0?'#08080d':'#0c0c14';
      return '<tr style="background:'+bg+'">'
        +row.map(function(cell,ci){
          var color=ci===1?C.accent:ci===2?C.blue:C.muted;
          return '<td style="padding:5px 8px;color:'+color+';border-bottom:1px solid '+C.border+';">'+cell+'</td>';
        }).join('')
        +'</tr>';
    }).join('')
    +'</table>'
  );

  out+=insight('🧪','Golden Rule',
    'Start simple. <span style="color:'+C.accent+';font-weight:700;">Linear Regression</span> is always your baseline — '
    +'if a more complex model cannot beat it by a meaningful margin, the complexity is not justified. '
    +'Use <span style="color:'+C.yellow+';font-weight:700;">cross-validation</span> to compare fairly, and '
    +'<span style="color:'+C.purple+';font-weight:700;">SHAP values</span> to explain whatever model you choose.'
  );
  return out;
}

/* ═══════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════ */
var TABS=[
  '📈 What is Regression?',
  '🗺️ Method Map (ASIC)',
  '📦 Family Breakdown',
  '🧭 When to Use'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.purple+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Regression</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Concept overview · all method families · ASIC diagram · decision guide')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  if(S.tab===0) html+=renderWhat();
  else if(S.tab===1) html+=renderASIC();
  else if(S.tab===2) html+=renderFamilies();
  else if(S.tab===3) html+=renderWhen();
  html+='</div>';
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
    if(tag==='button'||tag==='rect'||tag==='text'||tag==='polygon'){
      el.addEventListener('click',function(){
        if(action==='tab'){S.tab=idx;S.selectedFamily=-1;render();}
        else if(action==='selFamily'){S.selectedFamily=idx;render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='degree'){S.degree=val;render();}
        else if(action==='noise'){S.noise=val;render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

REG_VISUAL_HEIGHT = 1200